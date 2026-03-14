use crate::{KrigingError, Real};

const EARTH_RADIUS_KM: Real = 6_371.0;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GeoCoord {
    pub lat: Real,
    pub lon: Real,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct PreparedGeoCoord {
    lat_rad: Real,
    lon_rad: Real,
    cos_lat: Real,
}

impl GeoCoord {
    pub fn validate(&self) -> Result<(), KrigingError> {
        if !(-90.0..=90.0).contains(&self.lat) || !(-180.0..=180.0).contains(&self.lon) {
            return Err(KrigingError::InvalidCoordinate {
                lat: self.lat,
                lon: self.lon,
            });
        }
        Ok(())
    }
}

pub fn haversine_distance(coord1: GeoCoord, coord2: GeoCoord) -> Real {
    let prepared1 = prepare_geo_coord(coord1);
    let prepared2 = prepare_geo_coord(coord2);
    let a = haversine_a(prepared1, prepared2);
    let c = 2.0 * a.sqrt().atan2((1.0 - a).sqrt());
    EARTH_RADIUS_KM * c
}

pub(crate) fn prepare_geo_coord(coord: GeoCoord) -> PreparedGeoCoord {
    let lat_rad = coord.lat.to_radians();
    let lon_rad = coord.lon.to_radians();
    PreparedGeoCoord {
        lat_rad,
        lon_rad,
        cos_lat: lat_rad.cos(),
    }
}

pub(crate) fn haversine_distance_prepared(
    coord1: PreparedGeoCoord,
    coord2: PreparedGeoCoord,
) -> Real {
    let a = haversine_a(coord1, coord2);
    let c = 2.0 * a.sqrt().atan2((1.0 - a).sqrt());
    EARTH_RADIUS_KM * c
}

fn haversine_a(coord1: PreparedGeoCoord, coord2: PreparedGeoCoord) -> Real {
    let dlat = coord2.lat_rad - coord1.lat_rad;
    let dlon = coord2.lon_rad - coord1.lon_rad;
    (dlat * 0.5).sin().powi(2) + coord1.cos_lat * coord2.cos_lat * (dlon * 0.5).sin().powi(2)
}

pub fn distance_matrix(coords: &[GeoCoord]) -> Vec<Vec<Real>> {
    let n = coords.len();
    let mut matrix = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in (i + 1)..n {
            let d = haversine_distance(coords[i], coords[j]);
            matrix[i][j] = d;
            matrix[j][i] = d;
        }
    }
    matrix
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn geo_coord_validation_rejects_invalid_inputs() {
        let bad = GeoCoord {
            lat: 120.0,
            lon: 0.0,
        };
        assert!(bad.validate().is_err());
    }

    #[test]
    fn haversine_distance_matches_known_equator_segment() {
        let coord1 = GeoCoord { lat: 0.0, lon: 0.0 };
        let coord2 = GeoCoord { lat: 0.0, lon: 1.0 };
        let dist = haversine_distance(coord1, coord2);
        assert_relative_eq!(dist, 111.1949, epsilon = 0.05);
    }

    #[test]
    fn distance_matrix_is_symmetric_with_zero_diagonal() {
        let coords = vec![
            GeoCoord { lat: 0.0, lon: 0.0 },
            GeoCoord { lat: 0.0, lon: 1.0 },
            GeoCoord { lat: 1.0, lon: 1.0 },
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
    fn prepared_haversine_matches_public_haversine() {
        let coord1 = GeoCoord {
            lat: 10.123,
            lon: -45.987,
        };
        let coord2 = GeoCoord {
            lat: -5.456,
            lon: 120.789,
        };
        let direct = haversine_distance(coord1, coord2);
        let prepared =
            haversine_distance_prepared(prepare_geo_coord(coord1), prepare_geo_coord(coord2));
        assert_relative_eq!(direct, prepared, epsilon = 1e-6);
    }
}
