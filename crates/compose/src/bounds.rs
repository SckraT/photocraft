//! `Surface::content_bounds` with the per-tile scan cached by copy-on-write tile identity, so
//! layer bounds cost a hash lookup per tile instead of a pixel scan per render tile (the CPU
//! compositor asks for an effect layer's bounds in every tile; the GPU planner every frame).

use std::collections::HashMap;
use std::sync::{Arc, Mutex, Weak};

use photocraft_geom::{Rect, TileCoord};
use photocraft_raster::{Surface, Tile};

struct Entry {
    tile: Weak<Tile>,
    /// Content bounds inside the tile (tile-local), for one default pixel.
    default: Box<[u8]>,
    bounds: Option<(u16, u16, u16, u16)>,
}

/// The entries, and the size past which dead ones are dropped: twice what survived the last
/// sweep, so a cache full of live tiles isn't swept again on every call.
struct Cache {
    entries: HashMap<usize, Entry>,
    sweep_at: usize,
}

const MIN_SWEEP: usize = 1 << 16;

fn cache() -> &'static Mutex<Cache> {
    static C: std::sync::OnceLock<Mutex<Cache>> = std::sync::OnceLock::new();
    C.get_or_init(|| Mutex::new(Cache { entries: HashMap::new(), sweep_at: MIN_SWEEP }))
}

/// `Surface::content_bounds`, cached per tile.
pub fn content_bounds(s: &Surface) -> Rect {
    if s.tile_count() == 0 {
        return Rect::EMPTY;
    }
    let fmt = s.format();
    let mut dp = vec![0u8; fmt.bytes_per_pixel()];
    photocraft_raster::encode_pixel(&fmt, &s.default_pixel(), &mut dp);
    let mut out = Rect::EMPTY;
    let mut add = |coord: &TileCoord, b: Option<(u16, u16, u16, u16)>| {
        if let Some((x0, y0, x1, y1)) = b {
            let o = coord.rect();
            let r = Rect::new(o.x0 + x0 as i32, o.y0 + y0 as i32, o.x0 + x1 as i32, o.y0 + y1 as i32);
            out = if out.is_empty() { r } else { out.union(&r) };
        }
    };
    // Hits are read under the lock; misses are scanned without it, so compositor threads asking
    // for other layers' bounds don't queue behind a scan.
    let mut misses = Vec::new();
    {
        let mut c = cache().lock().unwrap_or_else(|e| e.into_inner());
        if c.entries.len() > c.sweep_at {
            c.entries.retain(|_, e| e.tile.strong_count() > 0);
            c.sweep_at = c.entries.len().saturating_mul(2).max(MIN_SWEEP);
        }
        for (coord, t) in s.tiles() {
            let hit = c.entries.get(&(Arc::as_ptr(t) as usize)).filter(|e| e.tile.upgrade().is_some_and(|u| Arc::ptr_eq(&u, t)) && *e.default == *dp);
            match hit {
                Some(e) => add(coord, e.bounds),
                None => misses.push((coord, t)),
            }
        }
    }
    if misses.is_empty() {
        return out;
    }
    let scanned: Vec<_> = misses.into_iter().map(|(coord, t)| (coord, t, t.content_bounds(&dp))).collect();
    let mut c = cache().lock().unwrap_or_else(|e| e.into_inner());
    for (coord, t, b) in scanned {
        c.entries.insert(Arc::as_ptr(t) as usize, Entry { tile: Arc::downgrade(t), default: dp.clone().into_boxed_slice(), bounds: b });
        add(coord, b);
    }
    out
}
