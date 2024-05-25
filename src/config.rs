use r2r::RosParams;

#[derive(RosParams, Debug)]
pub struct ConfigParse {
    pub verbose: bool,
    pub aabb_min_x: f32,
    pub aabb_max_x: f32,
    pub aabb_min_y: f32,
    pub aabb_max_y: f32,
    pub aabb_min_z: f32,
    pub aabb_max_z: f32,
    pub invert_aabb: bool,
    pub min_dist: f32,
    pub max_dist: f32,
    pub invert_distance: bool,
    pub global_zero_frame: String,
    pub invert_polygon: bool,
    pub global_frame: String,
    pub fov_right: f32,
    pub fov_left: f32,
    pub fov_enabled: bool,
    pub invert_fov: bool,
    pub disable_with_first_polygon: bool,
    pub min_intensity: f32,
    pub max_intensity: f32,
    pub invert_intensity: bool,
}

impl Default for ConfigParse {
    fn default() -> Self {
        ConfigParse {
            verbose: false,
            aabb_min_x: f32::MIN,
            aabb_max_x: f32::MAX,
            aabb_min_y: f32::MIN,
            aabb_max_y: f32::MAX,
            aabb_min_z: f32::MIN,
            aabb_max_z: f32::MAX,
            invert_aabb: false,
            min_dist: 0.0,
            max_dist: f32::MAX / 2.5 as f32,
            invert_distance: false,
            global_zero_frame: "".to_string(),
            invert_polygon: false,
            global_frame: "".to_string(),
            fov_right: 0.0,
            fov_left: 0.0,
            fov_enabled: false,
            invert_fov: false,
            disable_with_first_polygon: false,
            min_intensity: f32::MIN,
            max_intensity: f32::MAX,
            invert_intensity: false,
        }
    }
}
