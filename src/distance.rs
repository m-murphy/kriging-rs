use crate::{KrigingError, Real};

const EARTH_RADIUS_KM: Real = 6_371.0;

/// Geographic coordinate in degrees: latitude in [-90, 90], longitude in [-180, 180].
///
/// Use [`try_new`](Self::try_new) to construct; distances are computed with Haversine (km).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GeoCoord {
    lat: Real,
    lon: Real,
}

/// Cached trigonometric form of a [`GeoCoord`] for repeated Haversine evaluations.
///
/// Construct with [`prepare_geo_coord`]; fields are crate-internal because their meaning is
/// bound to the Haversine formula and not part of the stable API.
#[derive(Debug, Clone, Copy)]
pub struct PreparedGeoCoord {
    pub(crate) lat_rad: Real,
    pub(crate) lon_rad: Real,
    pub(crate) cos_lat: Real,
}

impl GeoCoord {
    /// Creates a valid coordinate. Returns an error if `lat` is not in [-90, 90] or `lon` is not in [-180, 180].
    pub fn try_new(lat: Real, lon: Real) -> Result<Self, KrigingError> {
        if !(-90.0..=90.0).contains(&lat) || !(-180.0..=180.0).contains(&lon) {
            return Err(KrigingError::InvalidCoordinate { lat, lon });
        }
        Ok(Self { lat, lon })
    }

    #[inline]
    pub fn lat(self) -> Real {
        self.lat
    }

    #[inline]
    pub fn lon(self) -> Real {
        self.lon
    }
}

pub fn prepare_geo_coord(coord: GeoCoord) -> PreparedGeoCoord {
    let lat_rad = coord.lat().to_radians();
    let lon_rad = coord.lon().to_radians();
    PreparedGeoCoord {
        lat_rad,
        lon_rad,
        cos_lat: lat_rad.cos(),
    }
}

pub fn haversine_distance(coord1: GeoCoord, coord2: GeoCoord) -> Real {
    let prepared1 = prepare_geo_coord(coord1);
    let prepared2 = prepare_geo_coord(coord2);
    let a = haversine_a(prepared1, prepared2);
    // Clamp (1 - a) against f32 round-off so near-antipodal pairs don't produce NaN; matches WGSL.
    let c = 2.0 * a.sqrt().atan2((1.0 - a).max(0.0).sqrt());
    EARTH_RADIUS_KM * c
}

pub fn haversine_distance_prepared(coord1: PreparedGeoCoord, coord2: PreparedGeoCoord) -> Real {
    let a = haversine_a(coord1, coord2);
    let c = 2.0 * a.sqrt().atan2((1.0 - a).max(0.0).sqrt());
    EARTH_RADIUS_KM * c
}

fn haversine_a(coord1: PreparedGeoCoord, coord2: PreparedGeoCoord) -> Real {
    let dlat = coord2.lat_rad - coord1.lat_rad;
    let dlon = coord2.lon_rad - coord1.lon_rad;
    (dlat * 0.5).sin().powi(2) + coord1.cos_lat * coord2.cos_lat * (dlon * 0.5).sin().powi(2)
}

/// Symmetric `n × n` distance matrix as a nested `Vec<Vec<Real>>`.
///
/// Kept for API compatibility. For hot paths prefer [`distance_matrix_flat`], which stores
/// distances in a single contiguous row-major buffer (`matrix[i * n + j]`) and is much more
/// cache-friendly — the nested version allocates `n + 1` separate heap blocks and scatters
/// rows across memory.
pub fn distance_matrix(coords: &[GeoCoord]) -> Vec<Vec<Real>> {
    let n = coords.len();
    let flat = distance_matrix_flat(coords);
    (0..n).map(|i| flat[i * n..(i + 1) * n].to_vec()).collect()
}

/// Flat row-major `n × n` distance matrix. Entry `(i, j)` is at index `i * n + j`.
///
/// Single contiguous allocation, symmetric, zero diagonal. Filled only over the upper
/// triangle to halve the Haversine calls.
pub fn distance_matrix_flat(coords: &[GeoCoord]) -> Vec<Real> {
    let n = coords.len();
    let mut flat = vec![0.0 as Real; n * n];
    for i in 0..n {
        for j in (i + 1)..n {
            let d = haversine_distance(coords[i], coords[j]);
            flat[i * n + j] = d;
            flat[j * n + i] = d;
        }
    }
    flat
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn geo_coord_try_new_rejects_invalid_inputs() {
        assert!(GeoCoord::try_new(120.0, 0.0).is_err());
        assert!(GeoCoord::try_new(0.0, 200.0).is_err());
    }

    #[test]
    fn haversine_distance_matches_known_equator_segment() {
        let coord1 = GeoCoord::try_new(0.0, 0.0).unwrap();
        let coord2 = GeoCoord::try_new(0.0, 1.0).unwrap();
        let dist = haversine_distance(coord1, coord2);
        assert_relative_eq!(dist, 111.1949, epsilon = 0.05);
    }

    #[test]
    fn distance_matrix_is_symmetric_with_zero_diagonal() {
        let coords = vec![
            GeoCoord::try_new(0.0, 0.0).unwrap(),
            GeoCoord::try_new(0.0, 1.0).unwrap(),
            GeoCoord::try_new(1.0, 1.0).unwrap(),
        ];
        let m = distance_matrix(&coords);
        assert_eq!(m[0][0], 0.0);
        assert_eq!(m[1][1], 0.0);
        assert_eq!(m[2][2], 0.0);
        assert_relative_eq!(m[0][1], m[1][0], epsilon = 1e-6);
        assert_relative_eq!(m[0][2], m[2][0], epsilon = 1e-6);
        assert_relative_eq!(m[1][2], m[2][1], epsilon = 1e-6);
    }

    #[test]
    fn distance_matrix_flat_matches_nested_layout() {
        let coords = vec![
            GeoCoord::try_new(0.0, 0.0).unwrap(),
            GeoCoord::try_new(0.0, 1.0).unwrap(),
            GeoCoord::try_new(1.0, 1.0).unwrap(),
        ];
        let nested = distance_matrix(&coords);
        let flat = distance_matrix_flat(&coords);
        assert_eq!(flat.len(), 9);
        for i in 0..3 {
            for j in 0..3 {
                assert_relative_eq!(nested[i][j], flat[i * 3 + j], epsilon = 1e-6);
            }
        }
    }

    #[test]
    fn prepared_haversine_matches_public_haversine() {
        let coord1 = GeoCoord::try_new(10.123, -45.987).unwrap();
        let coord2 = GeoCoord::try_new(-5.456, 120.789).unwrap();
        let direct = haversine_distance(coord1, coord2);
        let prepared =
            haversine_distance_prepared(prepare_geo_coord(coord1), prepare_geo_coord(coord2));
        assert_relative_eq!(direct, prepared, epsilon = 1e-6);
    }
}
