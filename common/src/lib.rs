
use lru::LruCache;
use parking_lot::{ Mutex, RwLock };
use std::cell::Cell;
use std::sync::{ Arc, atomic::AtomicBool };

pub use gyroflow_core::{ StabilizationManager, keyframes::*, stabilization::*, filesystem, gpu::* };
pub use gyroflow_core;

// re-exports
pub use rfd;
pub use parking_lot;
pub use lru;

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

pub type PluginResult<T> = Result<T, Box<dyn std::error::Error>>;

#[derive(Debug, Copy, Clone, Hash, PartialEq, PartialOrd, Eq, Ord, serde::Serialize, serde::Deserialize)]
pub enum Params {
    Logo,
    InstanceId,
    ProjectData,
    EmbeddedLensProfile,
    EmbeddedPreset,
    ProjectGroup, ProjectGroupEnd,
    LoadCurrent,
    ProjectPath,
    Browse,
    LoadLens,
    OpenGyroflow,
    ReloadProject,
    OpenRecentProject,
    Status,
    AdjustGroup, AdjustGroupEnd,
    Fov,
    Smoothness,
    ZoomLimit,
    LensCorrectionStrength,
    HorizonLockAmount,
    HorizonLockRoll,
    // PositionX,
    // PositionY,
    AdditionalPitch,
    AdditionalYaw,
    InputRotation,
    Rotation,
    VideoSpeed,
    DisableStretch,
    IntegrationMethod,
    KeyframesGroup, KeyframesGroupEnd,
    UseGyroflowsKeyframes,
    RecalculateKeyframes,
    OutputSizeGroup, OutputSizeGroupEnd,
    OutputWidth,
    OutputHeight,
    OutputSizeToTimeline,
    OutputSizeSwap,
    ToggleOverview,
    DontDrawOutside,
    IncludeProjectData,
    StabilizationSpeedRamp,
    InfoGroup, InfoGroupEnd,
    LoadedProject,
    LoadedPreset,
    LoadedLens,
    CreateCamera,
    Interpolation,
    FusionStartFrame,
    SaveDefault,
    ProjectAdvancedGroup, ProjectAdvancedGroupEnd,
    AdvancedGroup, AdvancedGroupEnd,
    QuickPreset,
}

thread_local! {
    pub static LOG_INITIALIZED: Cell<bool> = Cell::new(false);
}

pub struct GyroflowPluginBase {
    // We should cache managers globally because it's common to have the effect applied to the same clip and cut the clip into multiple pieces
    // We don't want to create a new manager for each piece of the same clip
    // Cache key is specific enough
    pub manager_cache: Mutex<ManagerCache>,

    pub context_initialized: bool,
}
impl Default for GyroflowPluginBase {
    fn default() -> Self {
        Self {
            manager_cache: Mutex::new(ManagerCache::default()),
            context_initialized: false,
        }
    }
}

pub struct ManagerCache {
    entries: LruCache<String, Arc<StabilizationManager>>,
    generation: u64,
}

impl Default for ManagerCache {
    fn default() -> Self {
        Self {
            entries: LruCache::new(std::num::NonZeroUsize::new(8).unwrap()),
            generation: 0,
        }
    }
}

impl std::ops::Deref for ManagerCache {
    type Target = LruCache<String, Arc<StabilizationManager>>;

    fn deref(&self) -> &Self::Target { &self.entries }
}

impl std::ops::DerefMut for ManagerCache {
    fn deref_mut(&mut self) -> &mut Self::Target { &mut self.entries }
}

fn resolve_cached_manager(
    key: &str,
    local: &mut LruCache<String, Arc<StabilizationManager>>,
    local_generation: &mut u64,
    shared: &mut ManagerCache,
) -> Option<Arc<StabilizationManager>> {
    if *local_generation != shared.generation {
        local.clear();
        *local_generation = shared.generation;
    }
    if let Some(stab) = shared.get(key).map(Arc::clone) {
        local.put(key.to_owned(), stab.clone());
        Some(stab)
    } else {
        local.get(key).map(Arc::clone)
    }
}

impl GyroflowPluginBase {
    /// If `disable_stretch` is true, inject a `plugin_disable_stretch` flag into gyroflow JSON data
    /// so that the setting persists when the data is embedded in a preset or project.
    fn maybe_inject_disable_stretch(data: &str, disable_stretch: bool) -> String {
        if !disable_stretch { return data.to_string(); }
        if let Ok(mut json) = serde_json::from_str::<serde_json::Value>(data) {
            json["plugin_disable_stretch"] = serde_json::Value::Bool(true);
            if let Ok(s) = serde_json::to_string(&json) {
                return s;
            }
        }
        data.to_string()
    }

    pub fn initialize_gpu_context(&mut self) {
        log::info!("GyroflowPluginBase::initialize_gpu_context");
        if !self.context_initialized {
            gyroflow_core::gpu::initialize_contexts();
            self.context_initialized = true;
        }
    }
    pub fn deinitialize_gpu_context(&mut self) {
        log::info!("GyroflowPluginBase::deinitialize_gpu_context");
    }

    pub fn initialize_log(&mut self, name: &str) {
        LOG_INITIALIZED.with(|x| {
            if !x.get() {
                log_panics::init();

                // #[cfg(target_os = "windows")] { win_dbg_logger::init(); }

                let tmp_log = std::env::temp_dir().join(format!("gyroflow-{name}.log"));

                let log_path = gyroflow_core::settings::data_dir().join(format!("gyroflow-{name}.log"));
                let log_config = [ "mp4parse", "wgpu", "naga", "akaze", "ureq", "rustls", "ofx" ]
                    .into_iter()
                    .fold(simplelog::ConfigBuilder::new(), |mut cfg, x| { cfg.add_filter_ignore_str(x); cfg })
                    .build();

                if let Ok(file_log) = std::fs::File::create(&log_path) {
                    let _ = simplelog::WriteLogger::init(log::LevelFilter::Debug, log_config, file_log);
                    x.set(true);
                } else if let Ok(file_log) = std::fs::File::create(&tmp_log) {
                    let _ = simplelog::WriteLogger::init(log::LevelFilter::Debug, log_config, file_log);
                    x.set(true);
                } else if cfg!(target_os = "linux") {
                    if let Ok(file_log) = std::fs::File::create(&format!("/tmp/gyroflow-{name}.log")) {
                        let _ = simplelog::WriteLogger::init(log::LevelFilter::Debug, log_config, file_log);
                        x.set(true);
                    } else {
                        eprintln!("Failed to create log file: {log_path:?}, {tmp_log:?}, /tmp/gyroflow-ofx.log");
                    }
                }
            }
        });
    }

    pub fn get_center_rect(width: usize, height: usize, org_ratio: f64) -> (usize, usize, usize, usize) {
        // If aspect ratio is different
        let new_ratio = width as f64 / height as f64;
        if (new_ratio - org_ratio).abs() > 0.1 {
            // Get center rect of original aspect ratio
            let rect = if new_ratio > org_ratio {
                ((height as f64 * org_ratio).round() as usize, height)
            } else {
                (width, (width as f64 / org_ratio).round() as usize)
            };
            (
                (width - rect.0) / 2, // x
                (height - rect.1) / 2, // y
                rect.0, // width
                rect.1 // height
            )
        } else {
            (0, 0, width, height)
        }
    }

    pub fn get_project_path(file_path: &str) -> Option<String> {
        let mut project_path = std::path::Path::new(file_path).with_extension("gyroflow");
        if !project_path.exists() {
            // Find first project path that begins with the file name
            if let Some(parent) = project_path.parent() {
                if let Ok(paths) = std::fs::read_dir(parent) {
                    if let Some(fname) = project_path.with_extension("").file_name().map(|x| x.to_string_lossy().to_string()) {
                        for path in paths {
                            if let Ok(path) = path {
                                let path_fname = path.file_name().to_string_lossy().to_string();
                                if path_fname.starts_with(&fname) && path_fname.ends_with(".gyroflow") {
                                    project_path = path.path();
                                    break;
                                }
                            }
                        }
                    }
                }
            }
        }
        if project_path.exists() {
            Some(project_path.to_string_lossy().to_string())
        } else {
            None
        }
    }

    pub fn get_gyroflow_location() -> Option<String> {
        match gyroflow_core::settings::try_get("exeLocation") {
            Some(serde_json::Value::String(v)) if !v.is_empty() => {
                Some(v)
            },
            _ => {
                if cfg!(target_os = "macos") && std::path::Path::new("/Applications/Gyroflow.app/Contents/MacOS/gyroflow").exists() {
                    Some("/Applications/Gyroflow.app".into())
                } else {
                    None
                }
            }
        }
    }

    pub fn open_gyroflow(project_path: Option<&str>) {
        if cfg!(target_os = "macos") {
            let mut cmd = std::process::Command::new("osascript");
            if let Some(project) = project_path {
                if !project.is_empty() {
                    cmd.args(&["-e", &format!("tell application \"Gyroflow\" to open file \"{}\"", project.replace("/", ":").trim_start_matches(':'))]);
                } else {
                    cmd.args(&["-e", "tell application \"Gyroflow\" to activate"]);
                }
            }
            let _ = cmd.output();
        } else {
            if let Some(v) = Self::get_gyroflow_location() {
                if !v.is_empty() {
                    if let Some(project) = project_path {
                        let result = if !project.is_empty() {
                            if cfg!(target_os = "macos") {
                                std::process::Command::new("open").args(["-a", &v, "--args", "--open", &project]).spawn()
                            } else if cfg!(target_os = "windows") && v.starts_with("shell:") {
                                let mut cmd = std::process::Command::new("cmd.exe");
                                #[cfg(target_os = "windows")]
                                { use std::os::windows::process::CommandExt; cmd.creation_flags(0x08000000); } // CREATE_NO_WINDOW
                                cmd.args(["/c", "start", "", &v, "--open", &project]).spawn()
                            } else {
                                std::process::Command::new(v).args(["--open", &project]).spawn()
                            }
                        } else {
                            if cfg!(target_os = "macos") {
                                std::process::Command::new("open").args(["-a", &v]).spawn()
                            } else if cfg!(target_os = "windows") && v.starts_with("shell:") {
                                let mut cmd = std::process::Command::new("cmd.exe");
                                #[cfg(target_os = "windows")]
                                { use std::os::windows::process::CommandExt; cmd.creation_flags(0x08000000); } // CREATE_NO_WINDOW
                                cmd.args(["/c", "start", "", &v]).spawn()
                            } else {
                                std::process::Command::new(v).spawn()
                            }
                        };
                        if let Err(e) = result {
                            rfd::MessageDialog::new().set_description(format!("Impossible de lancer Gyroflow : {e:?}")).show();
                        }
                    }
                }
            } else {
                rfd::MessageDialog::new().set_description("Application Gyroflow introuvable. Lance-la au moins une fois (version 1.6.0 ou plus récente), puis réessaie.").show();
            }
        }
    }

    pub fn get_param_definitions() -> [ParameterType; 15] {
        [
            ParameterType::HiddenString { id: "InstanceId" },
            ParameterType::HiddenString { id: "ProjectPath" },
            ParameterType::HiddenString { id: "ProjectData" },
            ParameterType::HiddenString { id: "EmbeddedLensProfile" },
            ParameterType::HiddenString { id: "EmbeddedPreset" },
            ParameterType::Group { id: "ProjectGroup", label: "Plan", opened: true, parameters: vec![
                ParameterType::Text    { id: "Status",            label: "État",                     hint: "État de la stabilisation pour ce plan. Survole ce champ pour le détail et la marche à suivre." },
                ParameterType::Button  { id: "LoadCurrent",       label: "Stabiliser ce plan",       hint: "Charge automatiquement les données gyro du plan (ou son projet .gyroflow s'il existe à côté du fichier) et lance la stabilisation." },
            ] },
            ParameterType::Group { id: "AdjustGroup", label: "Réglages", opened: true, parameters: vec![
                ParameterType::Select   { id: "QuickPreset",            label: "Préréglage rapide",    hint: "Applique d'un clic un jeu de réglages adapté au type de plan. Tu peux ensuite affiner chaque curseur.", options: vec!["Choisir…", "Marche / suivi", "Plan posé (quasi trépied)", "Horizon verrouillé", "Course / action"], default: "Choisir…" },
                ParameterType::Slider   { id: "Smoothness",             label: "Lissage",              hint: "Force de la stabilisation. Plus la valeur est haute, plus le mouvement est lissé (50 = équilibré, 150+ = quasi trépied).", min: 1.0,    max: 300.0, default: 50.0 },
                ParameterType::Slider   { id: "ZoomLimit",              label: "Zoom max (%)",         hint: "Zoom maximum autorisé pour masquer les bords noirs. Plus bas = plus d'image conservée mais une stabilisation moins forte.", min: 51.0,   max: 300.0, default: 100.0 },
                ParameterType::Slider   { id: "LensCorrectionStrength", label: "Correction d'objectif (%)", hint: "Corrige la déformation de l'objectif (grand angle). 100 = correction complète, 0 = image d'origine.", min: 0.0,    max: 100.0, default: 100.0 },
                ParameterType::Slider   { id: "HorizonLockAmount",      label: "Verrouillage horizon (%)", hint: "Garde l'horizon droit même si la caméra penche. 100 = horizon totalement verrouillé.", min: 0.0,    max: 100.0, default: 0.0 },
                ParameterType::Slider   { id: "HorizonLockRoll",        label: "Inclinaison horizon",  hint: "Ajuste l'angle de l'horizon verrouillé (en degrés), pour le redresser ou l'incliner volontairement.", min: -100.0, max: 100.0, default: 0.0 },
                ParameterType::Button   { id: "SaveDefault",            label: "Enregistrer comme défaut", hint: "Enregistre le Lissage, le Zoom max, la Correction d'objectif, l'Horizon et le FOV actuels : ils seront appliqués à chaque nouveau plan chargé." },
            ] },
            ParameterType::Group { id: "AdvancedGroup", label: "Avancé", opened: false, parameters: vec![
                ParameterType::Slider   { id: "AdditionalPitch",        label: "Décalage haut / bas",  hint: "Oriente l'image stabilisée vers le haut ou le bas (en degrés).", min: -180.0, max: 180.0, default: 0.0 },
                ParameterType::Slider   { id: "AdditionalYaw",          label: "Décalage gauche / droite", hint: "Oriente l'image stabilisée vers la gauche ou la droite (en degrés).", min: -180.0, max: 180.0, default: 0.0 },
                ParameterType::Slider   { id: "Rotation",               label: "Rotation de l'image",  hint: "Tourne l'image finale (en degrés).", min: -360.0, max: 360.0, default: 0.0 },
                ParameterType::Slider   { id: "InputRotation",          label: "Rotation du fichier source", hint: "Rotation enregistrée par la caméra (plans verticaux : 90 ou 270). Réglée automatiquement, n'y touche que si l'image est mal orientée.", min: -360.0, max: 360.0, default: 0.0 },
                ParameterType::Slider   { id: "Fov",                    label: "Champ de vision (FOV)", hint: "Élargit ou resserre le champ de vision après stabilisation. 1 = normal.", min: 0.1,    max: 3.0,   default: 1.0 },
                ParameterType::Slider   { id: "VideoSpeed",             label: "Vitesse vidéo (%)",    hint: "Change la vitesse du plan ici (ou anime-la avec des images clés) plutôt qu'avec le changement de vitesse de Resolve, pour garder la stabilisation synchronisée.", min: 0.0001, max: 1000.0, default: 100.0 },
                ParameterType::Checkbox { id: "DisableStretch",         label: "Désactiver l'étirement Gyroflow", hint: "À cocher seulement si tu as désanamorphosé le plan toi-même dans Resolve alors que le profil d'objectif Gyroflow l'étire déjà.", default: false },
                ParameterType::Select   { id: "IntegrationMethod",      label: "Calcul de l'orientation", hint: "Méthode de calcul de l'orientation à partir du gyro. VQF convient presque toujours. Essaie une autre méthode si l'horizon verrouillé dérive.", options: vec!["None", "Complementary", "VQF", "Simple gyro", "Simple gyro + accel", "Mahony", "Madgwick"], default: "VQF" },
            ] },
            ParameterType::Group { id: "ProjectAdvancedGroup", label: "Projet Gyroflow", opened: false, parameters: vec![
                ParameterType::Button  { id: "Browse",            label: "Ouvrir un projet…",          hint: "Choisir un fichier .gyroflow (ou une vidéo) à utiliser pour ce plan." },
                ParameterType::Button  { id: "LoadLens",          label: "Charger un profil / préréglage…", hint: "Choisir un profil d'objectif (.json) ou un préréglage Gyroflow (.gyroflow) à appliquer." },
                ParameterType::Button  { id: "OpenGyroflow",      label: "Ouvrir dans Gyroflow",       hint: "Ouvre ce plan dans l'application Gyroflow pour synchroniser ou régler finement, puis enregistre le projet et clique Recharger." },
                ParameterType::Button  { id: "ReloadProject",     label: "Recharger le projet",        hint: "Relit le projet .gyroflow après l'avoir modifié dans l'application Gyroflow." },
                ParameterType::Button  { id: "OpenRecentProject", label: "Dernier projet enregistré",  hint: "Charge le dernier projet enregistré dans l'application Gyroflow." },
            ] },
            ParameterType::Group { id: "KeyframesGroup", label: "Images clés", opened: false, parameters: vec![
                ParameterType::Checkbox { id: "UseGyroflowsKeyframes", label: "Utiliser les images clés de Gyroflow", hint: "Utilise les images clés créées dans l'application Gyroflow plutôt que celles de Resolve.", default: false },
                ParameterType::Checkbox { id: "StabilizationSpeedRamp",label: "Adapter la stabilisation à la vitesse", hint: "Lors d'une variation de vitesse, ajuste la force de stabilisation à la vitesse du plan.", default: true },
                ParameterType::Button   { id: "RecalculateKeyframes",  label: "Recalculer les images clés", hint: "Recalcule après avoir modifié les courbes (page Fusion)." },
                ParameterType::Button   { id: "CreateCamera",  label: "Créer une caméra", hint: "Créer un calque caméra" },
            ] },
            ParameterType::Group { id: "OutputSizeGroup", label: "Taille de sortie", opened: false, parameters: vec![
                ParameterType::Slider   { id: "OutputWidth",    label: "Largeur",  hint: "Largeur de l'image stabilisée (pixels)",  min: 1.0, max: 16384.0, default: 3840.0 },
                ParameterType::Slider   { id: "OutputHeight",   label: "Hauteur", hint: "Hauteur de l'image stabilisée (pixels)", min: 1.0, max: 16384.0, default: 2160.0 },
                ParameterType::Button   { id: "OutputSizeToTimeline", label: "Adapter à la timeline", hint: "Règle la taille de sortie sur celle de la timeline" },
                ParameterType::Button   { id: "OutputSizeSwap",  label: "Inverser largeur / hauteur", hint: "Échange la largeur et la hauteur (horizontal ↔ vertical)" },
                ParameterType::Select   { id: "Interpolation",   label: "Interpolation", hint: "Méthode de mise à l'échelle. Lanczos4 = la plus nette.", options: vec!["Lanczos4", "RobidouxSharp", "Bilinear", "Bicubic", "Robidoux", "Mitchell", "CatmullRom"], default: "Lanczos4" },
            ] },
            ParameterType::Checkbox { id: "ToggleOverview",     label: "Aperçu de la stabilisation", hint: "Dézoome l'affichage pour voir le cadre stabilisé et les bords. À DÉSACTIVER avant l'export.", default: false },
            ParameterType::Checkbox { id: "DontDrawOutside",    label: "Ne pas dessiner hors du plan", hint: "Si le format du plan et de la timeline diffèrent, garde l'image dans le cadre du plan d'origine.", default: true },
            ParameterType::Checkbox { id: "IncludeProjectData", label: "Intégrer les données au projet", hint: "Enregistre les données Gyroflow (gyro compris) dans le projet Resolve : utile pour partager le projet sans les fichiers .gyroflow, mais le projet devient plus lourd.", default: false },
            ParameterType::Group { id: "InfoGroup", label: "Infos", opened: true, parameters: vec![
                ParameterType::Text { id: "LoadedProject",      label: "Fichier chargé",      hint: "Projet ou vidéo chargé" },
                ParameterType::Text { id: "LoadedPreset",       label: "Préréglage chargé",   hint: "Préréglage chargé" },
                ParameterType::Text { id: "LoadedLens",         label: "Profil d'objectif",   hint: "Profil d'objectif chargé" },
            ] },
        ]
    }
}

pub enum ParameterType {
    HiddenString { id: &'static str },
    TextBox      { id: &'static str, label: &'static str, hint: &'static str },
    Text         { id: &'static str, label: &'static str, hint: &'static str },
    Slider       { id: &'static str, label: &'static str, hint: &'static str, min: f64, max: f64, default: f64 },
    Checkbox     { id: &'static str, label: &'static str, hint: &'static str, default: bool },
    Button       { id: &'static str, label: &'static str, hint: &'static str },
    Group        { id: &'static str, label: &'static str, opened: bool, parameters: Vec<ParameterType> },
    Select       { id: &'static str, label: &'static str, hint: &'static str, options: Vec<&'static str>, default: &'static str },
}

#[derive(Debug, Clone)]
pub enum TimeType {
    Frame(f64),
    Milliseconds(f64),
    Microseconds(i64),
    FrameOrMicrosecond((Option<f64>, Option<i64>))
}
pub trait GyroflowPluginParams {
    fn set_enabled(&mut self, param: Params, enabled: bool) -> PluginResult<()>;
    fn set_label(&mut self, param: Params, label: &str) -> PluginResult<()>;
    fn set_hint(&mut self, param: Params, hint: &str) -> PluginResult<()>;

    fn set_f64(&mut self, param: Params, value: f64) -> PluginResult<()>;
    fn get_f64(&self, param: Params) -> PluginResult<f64>;
    fn get_f64_at_time(&self, param: Params, time: TimeType) -> PluginResult<f64>;
    fn set_bool(&mut self, param: Params, value: bool) -> PluginResult<()>;
    fn get_bool(&self, param: Params) -> PluginResult<bool>;
    fn get_bool_at_time(&self, param: Params, time: TimeType) -> PluginResult<bool>;
    fn set_string(&mut self, param: Params, value: &str) -> PluginResult<()>;
    fn get_string(&self, param: Params) -> PluginResult<String>;
    fn set_i32(&mut self, param: Params, value: i32) -> PluginResult<()>;
    fn get_i32(&self, param: Params) -> PluginResult<i32>;

    fn is_keyframed(&self, param: Params) -> bool;
    fn get_keyframes(&self, param: Params) -> Vec<(TimeType, f64)>;
    fn clear_keyframes(&mut self, param: Params) -> PluginResult<()>;
    fn set_f64_at_time(&mut self, param: Params, time: TimeType, value: f64) -> PluginResult<()>;
}

#[derive(Default, Clone)]
pub struct KeyframableParams {
    pub use_gyroflows_keyframes: bool,
    pub cached_keyframes: KeyframeManager
}

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct GyroflowPluginBaseInstance {
    #[serde(skip)]
    pub keyframable_params: Arc<RwLock<KeyframableParams>>,

    #[serde(skip)]
    pub managers: LruCache<String, Arc<StabilizationManager>>,

    #[serde(skip)]
    pub manager_cache_generation: u64,

    pub reload_values_from_project: bool,

    pub original_video_size: (usize, usize),
    pub original_output_size: (usize, usize),
    pub timeline_size: (usize, usize),
    pub num_frames: usize,
    pub fps: f64,
    pub has_motion: bool,
    pub ever_changed: bool,
    pub cache_keyframes_every_frame: bool,
    pub framebuffer_inverted: bool,
    pub anamorphic_adjust_size: bool,
    pub always_set_input_rotation: bool,

    pub opencl_disabled: bool,
}
impl Clone for GyroflowPluginBaseInstance {
    fn clone(&self) -> Self {
        Self {
            managers:                       self.managers.clone(),
            manager_cache_generation:       self.manager_cache_generation,
            original_output_size:           self.original_output_size,
            original_video_size:            self.original_video_size,
            timeline_size:                  self.timeline_size,
            num_frames:                     self.num_frames,
            fps:                            self.fps,
            has_motion:                     self.has_motion,
            reload_values_from_project:     self.reload_values_from_project,
            ever_changed:                   self.ever_changed,
            opencl_disabled:                self.opencl_disabled,
            cache_keyframes_every_frame:    self.cache_keyframes_every_frame,
            framebuffer_inverted:           self.framebuffer_inverted,
            anamorphic_adjust_size:         self.anamorphic_adjust_size,
            always_set_input_rotation:      self.always_set_input_rotation,
            keyframable_params:             Arc::new(RwLock::new(self.keyframable_params.read().clone())),
        }
    }
}
impl Default for GyroflowPluginBaseInstance {
    fn default() -> Self {
        Self {
            managers:                       LruCache::new(std::num::NonZeroUsize::new(20).unwrap()),
            manager_cache_generation:       0,
            original_output_size:           (0, 0),
            original_video_size:            (0, 0),
            timeline_size:                  (0, 0),
            num_frames:                     0,
            fps:                            0.0,
            has_motion:                     false,
            reload_values_from_project:     true,
            ever_changed:                   false,
            opencl_disabled:                false,
            cache_keyframes_every_frame:    true,
            framebuffer_inverted:           false,
            anamorphic_adjust_size:         true,
            always_set_input_rotation:      false,
            keyframable_params: Arc::new(RwLock::new(KeyframableParams {
                use_gyroflows_keyframes:  false, // TODO param_set.parameter::<Bool>("UseGyroflowsKeyframes")?.get_value()?,
                cached_keyframes:         KeyframeManager::default()
            })),
        }
    }
}

impl GyroflowPluginBaseInstance {
    pub fn update_loaded_state(&mut self, params: &mut dyn GyroflowPluginParams, loaded: bool) {
        let _ = params.set_enabled(Params::Fov, loaded);
        let _ = params.set_enabled(Params::Smoothness, loaded);
        let _ = params.set_enabled(Params::ZoomLimit, loaded);
        let _ = params.set_enabled(Params::LensCorrectionStrength, loaded);
        let _ = params.set_enabled(Params::HorizonLockAmount, loaded);
        let _ = params.set_enabled(Params::HorizonLockRoll, loaded);
        //let _ = params.set_enabled(Params::PositionX, loaded);
        //let _ = params.set_enabled(Params::PositionY, loaded);
        let _ = params.set_enabled(Params::AdditionalPitch, loaded);
        let _ = params.set_enabled(Params::AdditionalYaw, loaded);
        let _ = params.set_enabled(Params::Rotation, loaded);
        let _ = params.set_enabled(Params::VideoSpeed, loaded);
        let _ = params.set_enabled(Params::DisableStretch, loaded);
        let _ = params.set_enabled(Params::IntegrationMethod, loaded);
        let _ = params.set_enabled(Params::ToggleOverview, loaded);
        let _ = params.set_enabled(Params::ReloadProject, loaded);
        let _ = params.set_enabled(Params::OutputWidth, loaded);
        let _ = params.set_enabled(Params::OutputHeight, loaded);
        let _ = params.set_enabled(Params::OutputSizeToTimeline, loaded);
        let _ = params.set_enabled(Params::OutputSizeSwap, loaded);
        let _ = params.set_string(Params::Status, if loaded { "OK" } else { "Aucun plan chargé : clique « Stabiliser ce plan »" });
        let _ = params.set_label(Params::OpenGyroflow, if loaded { "Ouvrir dans Gyroflow" } else { "Ouvrir Gyroflow" });
    }

    pub fn initialize_instance_id(&mut self, instance_id: &mut String) {
        if instance_id.is_empty() {
            self.ever_changed = true;
            *instance_id = format!("{}", fastrand::u64(..));
        }
    }

    pub fn set_keyframe_provider(&self, stab: &StabilizationManager) {
        let kparams = self.keyframable_params.clone();
        stab.keyframes.write().set_custom_provider(move |kf, typ, timestamp_ms| -> Option<f64> {
            let params = kparams.read();
            if params.use_gyroflows_keyframes && kf.is_keyframed_internally(typ) { return None; }
            params.cached_keyframes.value_at_video_timestamp(typ, timestamp_ms)
        });
    }
    pub fn cache_keyframes(&mut self, params: &dyn GyroflowPluginParams, use_gyroflows_keyframes: bool, num_frames: usize, fps: f64) {
        let mut mgr = KeyframeManager::new();
        macro_rules! cache_key {
            ($typ:expr, $param:expr, $scale:expr) => {
                if params.is_keyframed($param) {
                    log::info!("param: {:?} is keyframed, cache_keyframes_every_frame: {}", $param, self.cache_keyframes_every_frame);
                    if self.cache_keyframes_every_frame { // Query every frame
                        for t in 0..num_frames {
                            let time = t as f64;
                            let timestamp_us = ((time / fps * 1_000_000.0)).round() as i64;

                            if let Ok(v) = params.get_f64_at_time($param, TimeType::FrameOrMicrosecond((Some(time), Some(timestamp_us)))) {
                                mgr.set(&$typ, timestamp_us, v / $scale);
                            }
                        }
                    } else {
                        // Cache only the keyframes at their timestamps
                        for (t, v) in params.get_keyframes($param) {
                            let timestamp_us = match t {
                                TimeType::FrameOrMicrosecond((Some(f), None)) |
                                TimeType::Frame(f) => ((f / fps * 1_000_000.0)).round() as i64,
                                TimeType::Milliseconds(ms) => (ms * 1_000.0).round() as i64,
                                TimeType::Microseconds(us) => us,
                                TimeType::FrameOrMicrosecond((_,    Some(timestamp_us))) => timestamp_us,
                                TimeType::FrameOrMicrosecond((None, None)) => unreachable!(),
                            };

                            mgr.set(&$typ, timestamp_us, v / $scale);
                        }
                    }
                } else {
                    log::info!("param: {:?} NOT keyframed", $param);
                    if let Ok(v) = params.get_f64($param) {
                        mgr.set(&$typ, 0, v / $scale);
                    }
                }
            };
        }
        cache_key!(KeyframeType::Fov,                       Params::Fov,                    1.0);
        cache_key!(KeyframeType::MaxZoom,                   Params::ZoomLimit,              1.0);
        cache_key!(KeyframeType::SmoothingParamSmoothness,  Params::Smoothness,             100.0);
        cache_key!(KeyframeType::LensCorrectionStrength,    Params::LensCorrectionStrength, 100.0);
        cache_key!(KeyframeType::LockHorizonAmount,         Params::HorizonLockAmount,      1.0);
        cache_key!(KeyframeType::LockHorizonRoll,           Params::HorizonLockRoll,        1.0);
        cache_key!(KeyframeType::VideoSpeed,                Params::VideoSpeed,             100.0);
        cache_key!(KeyframeType::VideoRotation,             Params::Rotation,               1.0);
        //cache_key!(KeyframeType::ZoomingCenterX,            Params::PositionX,              100.0);
        //cache_key!(KeyframeType::ZoomingCenterY,            Params::PositionY,              100.0);
        cache_key!(KeyframeType::AdditionalRotationX,       Params::AdditionalYaw,          1.0);
        cache_key!(KeyframeType::AdditionalRotationY,       Params::AdditionalPitch,        1.0);

        let mut kparams = self.keyframable_params.write();
        kparams.use_gyroflows_keyframes = use_gyroflows_keyframes;
        kparams.cached_keyframes = mgr;
    }

    pub fn stab_manager(&mut self, params: &mut dyn GyroflowPluginParams, manager_cache: &Mutex<ManagerCache>, out_size: (usize, usize), open_gyroflow_if_no_data: bool) -> PluginResult<Arc<StabilizationManager>> {
        let mut disable_stretch = params.get_bool(Params::DisableStretch)?;

        let instance_id = params.get_string(Params::InstanceId)?;
        let path = params.get_string(Params::ProjectPath)?;
        if path.is_empty() {
            self.update_loaded_state(params, false);
            return Err("Path is empty".into());
        }

        if self.timeline_size == (0, 0) {
            self.timeline_size = out_size;
        }

        let key = format!("{path}{disable_stretch}{instance_id}");
        let cloned = {
            let mut cache = manager_cache.lock();
            resolve_cached_manager(
                &key,
                &mut self.managers,
                &mut self.manager_cache_generation,
                &mut cache,
            )
        };
        let stab = if let Some(stab) = cloned {
            self.set_keyframe_provider(&stab);
            stab
        } else {
            log::info!("new stab manager for key: {key}");
            let mut stab = StabilizationManager::default();
            {
                // Find first lens profile database with loaded profiles
                let lock = manager_cache.lock();
                for (_, v) in lock.iter() {
                    if v.lens_profile_db.read().loaded {
                        stab.lens_profile_db = v.lens_profile_db.clone();
                        break;
                    }
                }
            }
            {
                let mut stab = stab.stabilization.write();
                stab.share_wgpu_instances = true;
                stab.interpolation = match params.get_i32(Params::Interpolation) {
                    Ok(1) => gyroflow_core::stabilization::Interpolation::RobidouxSharp,
                    Ok(2) => gyroflow_core::stabilization::Interpolation::Bilinear,
                    Ok(3) => gyroflow_core::stabilization::Interpolation::Bicubic,
                    Ok(4) => gyroflow_core::stabilization::Interpolation::Robidoux,
                    Ok(5) => gyroflow_core::stabilization::Interpolation::Mitchell,
                    Ok(6) => gyroflow_core::stabilization::Interpolation::CatmullRom,
                    _     => gyroflow_core::stabilization::Interpolation::Lanczos4,
                };
                log::info!("Interpolation: {:?}", stab.interpolation);
            }

            if !path.ends_with(".gyroflow") {
                let url = filesystem::path_to_url(&path);
                let mut file = filesystem::open_file(&url, false, false)?;
                let filesize = file.size;
                match stab.load_video_file(file.get_file(), filesize, &url, None, true) {
                    Ok(md) => {
                        if out_size != (0, 0) {
                            stab.params.write().output_size = out_size; // Default to timeline output size
                        }
                        if let Some(preset_out_size) = stab.input_file.read().preset_output_size {
                            stab.params.write().output_size = preset_out_size;
                        }

                        if let Ok(d) = params.get_string(Params::EmbeddedLensProfile) {
                            if !d.is_empty() {
                                if let Err(e) = stab.load_lens_profile(&d) {
                                    rfd::MessageDialog::new()
                                        .set_description(&format!("Impossible de charger le profil d'objectif : {e:?}"))
                                        .show();
                                }
                            }
                        }
                        if let Ok(d) = params.get_string(Params::EmbeddedPreset) {
                            if !d.is_empty() {
                                let mut is_preset = false;
                                if let Err(e) = stab.import_gyroflow_data(d.as_bytes(), true, None, |_|(), Arc::new(AtomicBool::new(false)), &mut is_preset, true) {
                                    rfd::MessageDialog::new()
                                        .set_description(&format!("Impossible de charger le préréglage : {e:?}"))
                                        .show();
                                }
                            }
                        }
                        if params.get_bool(Params::IncludeProjectData)? {
                            if let Ok(data) = stab.export_gyroflow_data(gyroflow_core::GyroflowProjectType::WithGyroData, "{}", None) {
                                let data = GyroflowPluginBase::maybe_inject_disable_stretch(&data, disable_stretch);
                                params.set_string(Params::ProjectData, &data)?;
                            }
                        }
                        if md.rotation != 0 && self.reload_values_from_project {
                            let r = ((360 - md.rotation) % 360) as f64;
                            params.set_f64(Params::InputRotation, r)?;
                            stab.params.write().video_rotation = r;
                        }
                        params.set_string(Params::LoadedProject, &filesystem::get_filename(&filesystem::path_to_url(&path)))?;
                        if !stab.gyro.read().file_metadata.read().has_accurate_timestamps && open_gyroflow_if_no_data {
                            GyroflowPluginBase::open_gyroflow(params.get_string(Params::ProjectPath).ok().as_deref());
                        }
                    },
                    Err(e) => {
                        let embedded_data = params.get_string(Params::ProjectData)?;
                        if !embedded_data.is_empty() {
                            let mut is_preset = false;
                            stab.import_gyroflow_data(embedded_data.as_bytes(), true, None, |_|(), Arc::new(AtomicBool::new(false)), &mut is_preset, true).map_err(|e| {
                                self.update_loaded_state(params, false);
                                format!("load_gyro_data error: {e}")
                            })?;
                        } else {
                            log::error!("An error occured: {e:?}");
                            self.update_loaded_state(params, false);
                            params.set_string(Params::Status, "Impossible de lire ce fichier")?;
                            params.set_hint(Params::Status, &format!("Le fichier n'a pas pu être lu ({path}) : {e:?}. Vérifie que le disque est branché et que le fichier existe."))?;
                            if open_gyroflow_if_no_data {
                                GyroflowPluginBase::open_gyroflow(params.get_string(Params::ProjectPath).ok().as_deref());
                            }
                            return Err(e.into());
                        }
                    }
                }
            } else {
                let project_data = {
                    if params.get_bool(Params::IncludeProjectData)? && !params.get_string(Params::ProjectData)?.is_empty() {
                        params.get_string(Params::ProjectData)?
                    } else if let Ok(data) = std::fs::read_to_string(&path) {
                        if params.get_bool(Params::IncludeProjectData)? {
                            params.set_string(Params::ProjectData, &data)?;
                        } else {
                            params.set_string(Params::ProjectData, "")?;
                        }
                        data
                    } else {
                        "".to_string()
                    }
                };
                let mut is_preset = false;
                stab.import_gyroflow_data(project_data.as_bytes(), true, Some(&filesystem::path_to_url(&path)), |_|(), Arc::new(AtomicBool::new(false)), &mut is_preset, true).map_err(|e| {
                    self.update_loaded_state(params, false);
                    format!("load_gyro_data error: {e}")
                })?;
                params.set_string(Params::LoadedProject, &filesystem::get_filename(&filesystem::path_to_url(&path)))?;

                if self.always_set_input_rotation {
                    let url = stab.input_file.read().url.clone();
                    let mut file = filesystem::open_file(&url, false, false)?;
                    let filesize = file.size;
                    if let Ok(video_md) = gyroflow_core::util::get_video_metadata(file.get_file(), filesize, &url) {
                        if video_md.rotation != 0 && self.reload_values_from_project {
                            let r = ((360 - video_md.rotation) % 360) as f64;
                            params.set_f64(Params::InputRotation, r)?;
                            stab.params.write().video_rotation = r;
                        }
                    }
                }
            }

            let loaded = {
                stab.params.write().calculate_ramped_timestamps(&stab.keyframes.read(), false, true);
                let gf_params = stab.params.read();
                self.original_video_size = gf_params.size;
                self.original_output_size = gf_params.output_size;
                self.num_frames = gf_params.frame_count;
                self.fps = gf_params.fps;
                let loaded = gf_params.duration_ms > 0.0;
                if loaded && self.reload_values_from_project {
                    self.reload_values_from_project = false;
                    let smooth = stab.smoothing.read();
                    let smoothness = smooth.current().get_parameter("smoothness");
                    params.set_f64(Params::Fov,                    gf_params.fov)?;
                    params.set_f64(Params::Smoothness,             smoothness * 100.0)?;
                    params.set_f64(Params::ZoomLimit,              gf_params.max_zoom.unwrap_or(0.0))?;
                    params.set_f64(Params::LensCorrectionStrength, (gf_params.lens_correction_amount * 100.0).min(100.0))?;
                    params.set_f64(Params::HorizonLockAmount,      if smooth.horizon_lock.lock_enabled { smooth.horizon_lock.horizonlockpercent } else { 0.0 })?;
                    params.set_f64(Params::HorizonLockRoll,        if smooth.horizon_lock.lock_enabled { smooth.horizon_lock.horizonroll } else { 0.0 })?;
                    params.set_f64(Params::VideoSpeed,             gf_params.video_speed * 100.0)?;
                    //params.set_f64(Params::PositionX,              gf_params.adaptive_zoom_center_offset.0 * 100.0)?;
                    //params.set_f64(Params::PositionY,              gf_params.adaptive_zoom_center_offset.1 * 100.0)?;
                    params.set_f64(Params::AdditionalYaw,          gf_params.additional_rotation.0)?;
                    params.set_f64(Params::AdditionalPitch,        gf_params.additional_rotation.1)?;
                    params.set_f64(Params::Rotation,               gf_params.video_rotation)?;
                    params.set_i32(Params::IntegrationMethod,      stab.gyro.read().integration_method as i32)?;

                    params.set_f64(Params::OutputWidth,            self.original_output_size.0 as f64)?;
                    params.set_f64(Params::OutputHeight,           self.original_output_size.1 as f64)?;

                    params.set_i32(Params::Interpolation, match stab.stabilization.read().interpolation {
                        gyroflow_core::stabilization::Interpolation::Lanczos4      => 0,
                        gyroflow_core::stabilization::Interpolation::RobidouxSharp => 1,
                        gyroflow_core::stabilization::Interpolation::Bilinear      => 2,
                        gyroflow_core::stabilization::Interpolation::Bicubic       => 3,
                        gyroflow_core::stabilization::Interpolation::Robidoux      => 4,
                        gyroflow_core::stabilization::Interpolation::Mitchell      => 5,
                        gyroflow_core::stabilization::Interpolation::CatmullRom    => 6,
                    })?;

                    let keyframes = stab.keyframes.read();
                    let all_keys = keyframes.get_all_keys();
                    params.set_bool(Params::UseGyroflowsKeyframes, !all_keys.is_empty())?;
                    if let Some(name) = stab.input_file.read().preset_name.clone() {
                        params.set_string(Params::LoadedPreset, &name)?;
                    }
                    params.set_string(Params::LoadedLens, &stab.lens.read().get_display_name())?;

                    for k in all_keys {
                        if let Some(keys) = keyframes.get_keyframes(k) {
                            if !keys.is_empty() {
                                macro_rules! set_keys {
                                    ($name:expr, $scale:expr) => {
                                        params.clear_keyframes($name)?;
                                        for (ts, v) in keys {
                                            let ts = if k == &KeyframeType::VideoSpeed { gf_params.get_source_timestamp_at_ramped_timestamp(*ts) } else { *ts };
                                            let time = (((ts as f64 / 1000.0) * gf_params.fps) / 1000.0).round();
                                            params.set_f64_at_time($name, TimeType::Frame(time), v.value * $scale)?;
                                        }
                                    };
                                }
                                match k {
                                    KeyframeType::Fov                      => { set_keys!(Params::Fov,                    1.0); },
                                    KeyframeType::SmoothingParamSmoothness => { set_keys!(Params::Smoothness,             100.0); },
                                    KeyframeType::MaxZoom                  => { set_keys!(Params::ZoomLimit,              1.0); },
                                    KeyframeType::LensCorrectionStrength   => { set_keys!(Params::LensCorrectionStrength, 100.0); },
                                    KeyframeType::LockHorizonAmount        => { set_keys!(Params::HorizonLockAmount,      1.0); },
                                    KeyframeType::LockHorizonRoll          => { set_keys!(Params::HorizonLockRoll,        1.0); },
                                    KeyframeType::VideoSpeed               => { set_keys!(Params::VideoSpeed,             100.0); },
                                    KeyframeType::VideoRotation            => { set_keys!(Params::Rotation,               1.0); },
                                    //KeyframeType::ZoomingCenterX           => { set_keys!(Params::PositionX,              100.0); },
                                    //KeyframeType::ZoomingCenterY           => { set_keys!(Params::PositionY,              100.0); },
                                    KeyframeType::AdditionalRotationX      => { set_keys!(Params::AdditionalYaw,          1.0); },
                                    KeyframeType::AdditionalRotationY      => { set_keys!(Params::AdditionalPitch,        1.0); },
                                    _ => { }
                                }
                            }
                        }
                    }
                }
                let use_gyroflows_keyframes = params.get_bool(Params::UseGyroflowsKeyframes).unwrap_or_default();
                self.cache_keyframes(params, use_gyroflows_keyframes, self.num_frames, self.fps.max(1.0));
                self.has_motion = stab.gyro.read().has_motion();
                loaded
            };

            self.update_loaded_state(params, loaded);

            // Check if loaded preset/project/lens data contains the plugin_disable_stretch flag
            if !disable_stretch {
                let has_flag = |d: &str| -> bool {
                    serde_json::from_str::<serde_json::Value>(d).ok()
                        .and_then(|v| v.get("plugin_disable_stretch").and_then(|v| v.as_bool()))
                        .unwrap_or(false)
                };
                for param_id in [Params::EmbeddedLensProfile, Params::EmbeddedPreset, Params::ProjectData] {
                    if let Ok(d) = params.get_string(param_id) {
                        if !d.is_empty() && has_flag(&d) {
                            disable_stretch = true;
                            let _ = params.set_bool(Params::DisableStretch, true);
                            break;
                        }
                    }
                }
                // Also check default.gyroflow, which gyroflow-core auto-loads inside
                // load_video_file but never exposes through the plugin's Embedded* params.
                // Paths and fallback order match `StabilizationManager::load_video_file`.
                if !disable_stretch {
                    let paths = [
                        gyroflow_core::settings::data_dir().join("lens_profiles").join("default.gyroflow"),
                        gyroflow_core::lens_profile_database::LensProfileDatabase::get_path().join("default.gyroflow"),
                    ];
                    for path in &paths {
                        if let Ok(d) = std::fs::read_to_string(path) {
                            if has_flag(&d) {
                                disable_stretch = true;
                                let _ = params.set_bool(Params::DisableStretch, true);
                                break;
                            }
                            // Matches core: stop after the first existing path (settings preferred).
                            break;
                        }
                    }
                }
            }

            if disable_stretch {
                stab.disable_lens_stretch(self.anamorphic_adjust_size);
            }

            stab.set_fov_overview(params.get_bool(Params::ToggleOverview)?);

            {
                let mut params = stab.params.write();
                params.framebuffer_inverted = self.framebuffer_inverted;
            }

            stab.init_size();
            stab.set_output_size(params.get_f64(Params::OutputWidth)? as _, params.get_f64(Params::OutputHeight)? as _);

            self.set_keyframe_provider(&stab);

            if let Ok(im) = params.get_i32(Params::IntegrationMethod) {
                let mut gyro = stab.gyro.write();
                gyro.integration_method = im as usize;
                gyro.apply_transforms();
            }

            stab.invalidate_smoothing();
            stab.recompute_blocking();
            let inverse = !(params.get_bool(Params::UseGyroflowsKeyframes)? && stab.keyframes.read().is_keyframed_internally(&KeyframeType::VideoSpeed));
            stab.params.write().calculate_ramped_timestamps(&stab.keyframes.read(), inverse, inverse);

            let stab = Arc::new(stab);
            // Insert to static global cache
            manager_cache.lock().put(key.to_owned(), stab.clone());
            // Cache it in this instance as well
            self.managers.put(key.to_owned(), stab.clone());

            stab
        };

        Ok(stab)
    }

    pub fn clear_stab(&mut self, manager_cache: &Mutex<ManagerCache>) {
        let local_keys = self.managers.iter().map(|x| x.0.clone()).collect::<Vec<_>>();
        self.managers.clear();

        // If there are no more local references, delete it from global cache
        let mut lock = manager_cache.lock();
        for key in local_keys {
            if let Some(v) = lock.get(&key) {
                if Arc::strong_count(v) == 1 {
                    lock.pop(&key);
                }
            }
        }
    }

    pub fn disable_opencl(&mut self) {
        if !self.opencl_disabled {
            unsafe { std::env::set_var("NO_OPENCL", "1") };
            self.opencl_disabled = true;
        }
    }

    pub fn set_status(&mut self, params: &mut dyn GyroflowPluginParams, status: &str, hint: &str, ok: bool) {
        if params.get_string(Params::Status).unwrap_or_default() != status {
            let _ = params.set_string(Params::Status, status);
            let _ = params.set_hint(Params::Status, hint);
            if ok {
                self.update_loaded_state(params, ok);
            }
        }
    }

    pub fn browse(current_path: &str) -> String {
        let mut d = rfd::FileDialog::new()
            .add_filter("Project and video files", &["mp4", "mov", "mxf", "braw", "r3d", "insv", "gyroflow"]);
        if !current_path.is_empty() {
            if let Some(path) = std::path::Path::new(current_path).parent() {
                d = d.set_directory(path);
            }
        }
        if let Some(d) = d.pick_file() {
            d.display().to_string()
        } else {
            String::new()
        }
    }

    pub fn param_changed(&mut self, params: &mut dyn GyroflowPluginParams, manager_cache: &Mutex<ManagerCache>, param: Params, user_edited: bool) -> Result<(), Box<dyn std::error::Error>> {
        if param == Params::Browse {
            let new_path = Self::browse(&params.get_string(Params::ProjectPath)?);
            if !new_path.is_empty() {
                params.set_string(Params::ProjectPath, &new_path)?;
                self.reload_values_from_project = true;
            }
        }
        if param == Params::LoadLens {
            let lens_directory = gyroflow_core::settings::data_dir().join("lens_profiles");
            log::info!("lens directory: {lens_directory:?}");

            let mut d = rfd::FileDialog::new().add_filter("Lens profiles and presets", &["json", "gyroflow"]);
            d = d.set_directory(lens_directory);
            if let Some(d) = d.pick_file() {
                let d = d.display().to_string();
                if !d.is_empty() {
                    if let Ok(contents) = std::fs::read_to_string(&d) {
                        if d.ends_with(".json") {
                            params.set_string(Params::EmbeddedLensProfile, &contents)?;
                        } else {
                            params.set_string(Params::EmbeddedPreset, &contents)?;
                        }
                        self.reload_values_from_project = true;
                    }
                    self.clear_stab(&manager_cache);
                }
            }
        }
        if param == Params::QuickPreset && user_edited {
            // (lissage, zoom max, verrouillage horizon)
            let preset = match params.get_i32(Params::QuickPreset).unwrap_or(0) {
                1 => Some((50.0, 130.0, 0.0)),   // Marche / suivi
                2 => Some((180.0, 130.0, 100.0)), // Plan posé (quasi trépied)
                3 => Some((50.0, 130.0, 100.0)),  // Horizon verrouillé
                4 => Some((25.0, 150.0, 0.0)),    // Course / action
                _ => None,
            };
            if let Some((smooth, zoom, horizon)) = preset {
                params.set_f64(Params::Smoothness, smooth)?;
                params.set_f64(Params::ZoomLimit, zoom)?;
                params.set_f64(Params::HorizonLockAmount, horizon)?;
                // Recalculate the stabilization with the new values
                self.param_changed(params, manager_cache, Params::Smoothness, true)?;
            }
        }
        if param == Params::SaveDefault {
            let result = (|| -> Result<std::path::PathBuf, Box<dyn std::error::Error>> {
                let mut stabilization = serde_json::json!({
                    "fov":                    params.get_f64(Params::Fov)?,
                    "max_zoom":               params.get_f64(Params::ZoomLimit)?,
                    "lens_correction_amount": (params.get_f64(Params::LensCorrectionStrength)? / 100.0).clamp(0.0, 1.0),
                    "smoothing_params":       [{ "name": "smoothness", "value": params.get_f64(Params::Smoothness)? / 100.0 }],
                    "horizon_lock_amount":    params.get_f64(Params::HorizonLockAmount)?,
                    "horizon_lock_roll":      params.get_f64(Params::HorizonLockRoll)?,
                });
                // Keep the smoothing algorithm currently used by the loaded clip
                if let Some((_, m)) = self.managers.iter().next() {
                    if let Ok(data) = m.export_gyroflow_data(gyroflow_core::GyroflowProjectType::Simple, "{}", None) {
                        if let Ok(json) = serde_json::from_str::<serde_json::Value>(&data) {
                            if let Some(method) = json.get("stabilization").and_then(|x| x.get("method")).cloned() {
                                stabilization["method"] = method;
                            }
                        }
                    }
                }
                let preset = serde_json::json!({ "title": "Plugin default", "version": 4, "videofile": "", "stabilization": stabilization });

                let dir = gyroflow_core::settings::data_dir().join("lens_profiles");
                std::fs::create_dir_all(&dir)?;
                let path = dir.join("default.gyroflow");
                if path.exists() {
                    let ts = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|x| x.as_secs()).unwrap_or(0);
                    let _ = std::fs::copy(&path, dir.join(format!("default.gyroflow.bak-{ts}")));
                }
                std::fs::write(&path, serde_json::to_string_pretty(&preset)?)?;
                Ok(path)
            })();
            match result {
                Ok(path) => {
                    log::info!("Saved default preset to {path:?}");
                    rfd::MessageDialog::new().set_description(format!("Réglages par défaut enregistrés.\nIls seront appliqués au prochain plan chargé.\n\n{}", path.display())).show();
                },
                Err(e) => {
                    log::error!("Failed to save default preset: {e:?}");
                    rfd::MessageDialog::new().set_description(format!("Impossible d'enregistrer les réglages par défaut : {e}")).show();
                }
            }
        }
        if param == Params::OpenGyroflow {
            GyroflowPluginBase::open_gyroflow(params.get_string(Params::ProjectPath).ok().as_deref());
        }
        if param == Params::OpenRecentProject {
            let last_project = gyroflow_core::settings::get_str("lastProject", "");
            if !last_project.is_empty() {
                params.set_string(Params::ProjectPath, &last_project)?;
            }
        }
        if param == Params::ProjectPath || param == Params::ReloadProject || param == Params::DontDrawOutside {
            if param == Params::ProjectPath || param == Params::ReloadProject {
                self.reload_values_from_project = true;
            }
            self.clear_stab(&manager_cache);
        }
        if param == Params::IncludeProjectData {
            let path = params.get_string(Params::ProjectPath)?;
            let ds = params.get_bool(Params::DisableStretch).unwrap_or(false);
            if params.get_bool(Params::IncludeProjectData).unwrap_or_default() {
                if path.ends_with(".gyroflow") {
                    if let Ok(data) = std::fs::read_to_string(&path) {
                        if StabilizationManager::project_has_motion_data(data.as_bytes()) {
                            let data = GyroflowPluginBase::maybe_inject_disable_stretch(&data, ds);
                            params.set_string(Params::ProjectData, &data)?;
                        } else {
                            if let Some((_, stab)) = self.managers.peek_lru() {
                                if let Ok(data) = stab.export_gyroflow_data(gyroflow_core::GyroflowProjectType::WithGyroData, "{}", None) {
                                    let data = GyroflowPluginBase::maybe_inject_disable_stretch(&data, ds);
                                    params.set_string(Params::ProjectData, &data)?;
                                }
                            }
                        }
                    } else {
                        params.set_string(Params::ProjectData, "")?;
                    }
                } else {
                    if let Some((_, stab)) = self.managers.peek_lru() {
                        if let Ok(data) = stab.export_gyroflow_data(gyroflow_core::GyroflowProjectType::WithGyroData, "{}", None) {
                            let data = GyroflowPluginBase::maybe_inject_disable_stretch(&data, ds);
                            params.set_string(Params::ProjectData, &data)?;
                        }
                    }
                }
            } else {
                params.set_string(Params::ProjectData, &"")?;
            }
        }
        if user_edited {
            if param == Params::OutputWidth || param == Params::OutputHeight || param == Params::OutputSizeSwap || param == Params::OutputSizeToTimeline {
                if param == Params::OutputSizeSwap {
                    let (w, h) = (params.get_f64(Params::OutputWidth)?, params.get_f64(Params::OutputHeight)? as _);
                    params.set_f64(Params::OutputWidth, h)?;
                    params.set_f64(Params::OutputHeight, w)?;
                }
                if param == Params::OutputSizeToTimeline {
                    params.set_f64(Params::OutputWidth, self.timeline_size.0 as f64)?;
                    params.set_f64(Params::OutputHeight, self.timeline_size.1 as f64)?;
                }
                for (_, v) in self.managers.iter_mut() {
                    v.set_output_size(params.get_f64(Params::OutputWidth)? as _, params.get_f64(Params::OutputHeight)? as _);
                    v.invalidate_blocking_zooming();
                }
            }
            match param {
                Params::Fov | Params::Smoothness | Params::ZoomLimit | Params::LensCorrectionStrength |
                Params::HorizonLockAmount | Params::HorizonLockRoll |
                //Params::PositionX | Params::PositionY |
                Params::AdditionalPitch | Params::AdditionalYaw |
                Params::Rotation | Params::InputRotation | Params::VideoSpeed | Params::IntegrationMethod |
                Params::UseGyroflowsKeyframes | Params::RecalculateKeyframes => {

                    params.set_string(Params::Status, "Calcul en cours…")?;
                    if !self.ever_changed {
                        self.ever_changed = true;
                        params.set_string(Params::InstanceId, &format!("{}", fastrand::u64(..)))?;
                        self.clear_stab(manager_cache);
                    }
                    let use_gyroflows_keyframes = params.get_bool(Params::UseGyroflowsKeyframes).unwrap_or_default();
                    self.cache_keyframes(params, use_gyroflows_keyframes, self.num_frames, self.fps.max(1.0));
                    for (_, v) in self.managers.iter_mut() {
                        match param {
                            Params::IntegrationMethod => {
                                if let Ok(im) = params.get_i32(Params::IntegrationMethod) {
                                    let mut gyro = v.gyro.write();
                                    gyro.integration_method = im as usize;
                                    gyro.apply_transforms();
                                }
                                v.invalidate_blocking_smoothing();
                                v.invalidate_blocking_zooming();
                            }
                            Params::Smoothness | Params::ZoomLimit | Params::HorizonLockAmount | Params::HorizonLockRoll |
                            Params::AdditionalPitch | Params::AdditionalYaw | Params::RecalculateKeyframes => {
                                v.invalidate_blocking_smoothing();
                                v.invalidate_blocking_zooming();
                            },
                            //Params::PositionX | Params::PositionY |
                            Params::LensCorrectionStrength | Params::Rotation => {
                                v.invalidate_blocking_zooming();
                            },
                            _ => { }
                        }
                        v.invalidate_blocking_undistortion();
                        match param {
                            Params::VideoSpeed | Params::UseGyroflowsKeyframes | Params::RecalculateKeyframes => {
                                let inverse = !(use_gyroflows_keyframes && v.keyframes.read().is_keyframed_internally(&KeyframeType::VideoSpeed));
                                v.params.write().calculate_ramped_timestamps(&v.keyframes.read(), inverse, inverse);
                            },
                            _ => { }
                        }
                    }
                    params.set_string(Params::Status, "OK")?;
                },
                _ => { }
            }
            if param == Params::ToggleOverview {
                let on = params.get_bool(Params::ToggleOverview)?;
                for (_, v) in self.managers.iter_mut() {
                    v.set_fov_overview(on);
                    v.invalidate_blocking_undistortion();
                }
            }
            if param == Params::Interpolation {
                self.managers.clear();
                let mut cache = manager_cache.lock();
                cache.clear();
                cache.generation = cache.generation.wrapping_add(1);
                self.manager_cache_generation = cache.generation;
            }
        }

        Ok(())
    }
}

pub fn hash_string(s: &str) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    s.hash(&mut hasher);
    hasher.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn local_cache() -> LruCache<String, Arc<StabilizationManager>> {
        LruCache::new(std::num::NonZeroUsize::new(20).unwrap())
    }

    #[test]
    fn reuses_local_manager_after_shared_lru_eviction() {
        let mut shared = ManagerCache::default();
        let mut local = local_cache();
        let mut generation = 0;
        let first = Arc::new(StabilizationManager::default());
        shared.put("first".into(), first.clone());
        local.put("first".into(), first.clone());
        for index in 0..8 {
            shared.put(format!("other-{index}"), Arc::new(StabilizationManager::default()));
        }

        let resolved = resolve_cached_manager("first", &mut local, &mut generation, &mut shared).unwrap();

        assert!(Arc::ptr_eq(&first, &resolved));
    }

    #[test]
    fn rejects_local_manager_after_shared_invalidation() {
        let mut shared = ManagerCache::default();
        let mut local = local_cache();
        let mut generation = 0;
        local.put("clip".into(), Arc::new(StabilizationManager::default()));
        shared.generation += 1;

        assert!(resolve_cached_manager("clip", &mut local, &mut generation, &mut shared).is_none());
        assert!(!local.contains("clip"));
        assert_eq!(generation, shared.generation);
    }

    #[test]
    fn shared_manager_replaces_stale_local_manager() {
        let mut shared = ManagerCache::default();
        let mut local = local_cache();
        let mut generation = 0;
        let stale = Arc::new(StabilizationManager::default());
        let current = Arc::new(StabilizationManager::default());
        local.put("clip".into(), stale);
        shared.put("clip".into(), current.clone());

        let resolved = resolve_cached_manager("clip", &mut local, &mut generation, &mut shared).unwrap();

        assert!(Arc::ptr_eq(&current, &resolved));
        assert!(Arc::ptr_eq(&current, local.get("clip").unwrap()));
    }
}

impl std::str::FromStr for Params {
    type Err = serde_json::Error;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        serde_json::from_str(&format!("\"{}\"", s))
    }
}
impl ToString for Params {
    fn to_string(&self) -> String {
        format!("{:?}", self)
    }
}

#[macro_export]
macro_rules! define_params {
    ($name:ident {
        strings: [ $($str_enum:ident  => $str_field:ident: $str_host_type:ty,)* ],
        bools:   [ $($bool_enum:ident => $bool_field:ident: $bool_host_type:ty,)* ],
        f64s:    [ $($f64_enum:ident  => $f64_field:ident: $f64_host_type:ty,)* ],
        i32s:    [ $($i32_enum:ident  => $i32_field:ident: $i32_host_type:ty,)* ],

        get_string:  $gstr_s:ident   $gstr_p:ident                    $gstr_block:block,
        set_string:  $sstr_s:ident   $sstr_p:ident,   $sstr_v:ident   $sstr_block:block,
        get_bool:    $gbool_s:ident  $gbool_p:ident                   $gbool_block:block,
        set_bool:    $sbool_s:ident  $sbool_p:ident,  $sbool_v:ident  $sbool_block:block,
        get_f64:     $gf64_s:ident   $gf64_p:ident                    $gf64_block:block,
        set_f64:     $sf64_s:ident   $sf64_p:ident,   $sf64_v:ident   $sf64_block:block,
        get_i32:     $gi32_s:ident   $gi32_p:ident                    $gi32_block:block,
        set_i32:     $si32_s:ident   $si32_p:ident,   $si32_v:ident   $si32_block:block,
        set_label:   $slabel_s:ident $slabel_p:ident, $slabel_v:ident $slabel_block:block,
        set_hint:    $shint_s:ident  $shint_p:ident,  $shint_v:ident  $shint_block:block,
        set_enabled: $sen_s:ident    $sen_p:ident,    $sen_v:ident    $sen_block:block,
        get_bool_at_time: $gtbool_s:ident  $gtbool_p:ident, $gtbool_t:ident                $gtbool_block:block,
        get_f64_at_time:  $gtf64_s:ident   $gtf64_p:ident,  $gtf64_t:ident                 $gtf64_block:block,
        set_f64_at_time:  $stf64_s:ident  $stf64_p:ident,  $stf64_t:ident, $stf64_v:ident $stf64_block:block,
        is_keyframed: $iskeyframe_s:ident  $iskeyframe_p:ident $iskeyframe_block:block,
        get_keyframes: $gkeyframes_s:ident $gkeyframes_p:ident $gkeyframes_block:block,
        clear_keyframes: $clr_s:ident      $clr_p:ident $clr_block:block,

        $($additional_fields:ident: $additional_fields_t:ty,)*
    }) => {
        #[derive(Default)]
        pub struct ParamsAdditionalFields {
            $( pub $additional_fields: $additional_fields_t, )*
        }
        pub struct $name {
            $( $str_field: $str_host_type, )*
            $( $bool_field: $bool_host_type, )*
            $( $f64_field: $f64_host_type, )*
            $( $i32_field: $i32_host_type, )*

            pub fields: ParamsAdditionalFields,
        }
        impl GyroflowPluginParams for $name {
            fn get_string(&self, param: Params) -> $crate::PluginResult<String> {
                let $gstr_s = &self.fields;
                match param {
                    $( Params::$str_enum => { let $gstr_p = &self.$str_field; $gstr_block }, )*
                    _ => panic!("Wrong parameter type"),
                }
            }
            fn set_string(&mut self, param: Params, value: &str) -> $crate::PluginResult<()> {
                let mut $sstr_s = &mut self.fields;
                match param {
                    $( Params::$str_enum => { let $sstr_p = &mut self.$str_field; let $sstr_v = value; $sstr_block }, )*
                    _ => panic!("Wrong parameter type"),
                }
            }
            fn get_bool(&self, param: Params) -> $crate::PluginResult<bool> {
                let $gbool_s = &self.fields;
                match param {
                    $( Params::$bool_enum => { let $gbool_p = &self.$bool_field; $gbool_block }, )*
                    _ => panic!("Wrong parameter type"),
                }
            }
            fn set_bool(&mut self, param: Params, value: bool) -> $crate::PluginResult<()> {
                let mut $sbool_s = &mut self.fields;
                match param {
                    $( Params::$bool_enum => { let $sbool_p = &mut self.$bool_field; let $sbool_v = value; $sbool_block }, )*
                    _ => panic!("Wrong parameter type"),
                }
            }
            fn get_f64(&self, param: Params) -> $crate::PluginResult<f64> {
                let $gf64_s = &self.fields;
                match param {
                    $( Params::$f64_enum => { let $gf64_p = &self.$f64_field; $gf64_block }, )*
                    _ => panic!("Wrong parameter type"),
                }
            }
            fn set_f64(&mut self, param: Params, value: f64) -> $crate::PluginResult<()> {
                let mut $sf64_s = &mut self.fields;
                match param {
                    $( Params::$f64_enum => { let $sf64_p = &mut self.$f64_field; let $sf64_v = value; $sf64_block }, )*
                    _ => panic!("Wrong parameter type"),
                }
            }
            fn get_i32(&self, param: Params) -> $crate::PluginResult<i32> {
                let $gi32_s = &self.fields;
                match param {
                    $( Params::$i32_enum => { let $gi32_p = &self.$i32_field; $gi32_block }, )*
                    _ => panic!("Wrong parameter type"),
                }
            }
            fn set_i32(&mut self, param: Params, value: i32) -> $crate::PluginResult<()> {
                let mut $si32_s = &mut self.fields;
                match param {
                    $( Params::$i32_enum => { let $si32_p = &mut self.$i32_field; let $si32_v = value; $si32_block }, )*
                    _ => panic!("Wrong parameter type"),
                }
            }
            fn set_label(&mut self, param: Params, label: &str) -> $crate::PluginResult<()> {
                let mut $slabel_s = &mut self.fields;
                let $slabel_v = label;
                match param {
                    $( Params::$str_enum  => { let $slabel_p = &mut self.$str_field;  $slabel_block }, )*
                    $( Params::$bool_enum => { let $slabel_p = &mut self.$bool_field; $slabel_block }, )*
                    $( Params::$f64_enum  => { let $slabel_p = &mut self.$f64_field;  $slabel_block }, )*
                    $( Params::$i32_enum  => { let $slabel_p = &mut self.$i32_field;  $slabel_block }, )*
                    _ => panic!("Wrong parameter type"),
                }
            }
            fn set_hint(&mut self, param: Params, hint: &str) -> $crate::PluginResult<()> {
                let mut $shint_s = &mut self.fields;
                let $shint_v = hint;
                match param {
                    $( Params::$str_enum  => { let $shint_p = &mut self.$str_field;  $shint_block }, )*
                    $( Params::$bool_enum => { let $shint_p = &mut self.$bool_field; $shint_block }, )*
                    $( Params::$f64_enum  => { let $shint_p = &mut self.$f64_field;  $shint_block }, )*
                    $( Params::$i32_enum  => { let $shint_p = &mut self.$i32_field;  $shint_block }, )*
                    _ => panic!("Wrong parameter type"),
                }
            }
            fn set_enabled(&mut self, param: Params, enabled: bool) -> $crate::PluginResult<()> {
                let mut $sen_s = &mut self.fields;
                let $sen_v = enabled;
                match param {
                    $( Params::$str_enum  => { let $sen_p = &mut self.$str_field;  $sen_block }, )*
                    $( Params::$bool_enum => { let $sen_p = &mut self.$bool_field; $sen_block }, )*
                    $( Params::$f64_enum  => { let $sen_p = &mut self.$f64_field;  $sen_block }, )*
                    $( Params::$i32_enum  => { let $sen_p = &mut self.$i32_field;  $sen_block }, )*
                    _ => panic!("Wrong parameter type"),
                }
            }
            fn get_f64_at_time(&self, param: Params, time: TimeType) -> $crate::PluginResult<f64> {
                let $gtf64_s = &self.fields;
                match param {
                    $( Params::$f64_enum => { let $gtf64_p = &self.$f64_field; let $gtf64_t = time; $gtf64_block }, )*
                    _ => panic!("Wrong parameter type"),
                }
            }
            fn get_bool_at_time(&self, param: Params, time: TimeType) -> $crate::PluginResult<bool> {
                let $gtbool_s = &self.fields;
                match param {
                    $( Params::$bool_enum => { let $gtbool_p = &self.$bool_field; let $gtbool_t = time; $gtbool_block }, )*
                    _ => panic!("Wrong parameter type"),
                }
            }
            fn clear_keyframes(&mut self, param: Params) -> $crate::PluginResult<()> {
                let mut $clr_s = &mut self.fields;
                match param {
                    $( Params::$f64_enum => { let $clr_p = &mut self.$f64_field; $clr_block }, )*
                    _ => panic!("Wrong parameter type"),
                }
            }
            fn is_keyframed(&self, param: Params) -> bool {
                let $iskeyframe_s = &self.fields;
                match param {
                    $( Params::$f64_enum => { let $iskeyframe_p = &self.$f64_field; $iskeyframe_block }, )*
                    _ => panic!("Wrong parameter type"),
                }
            }
            fn get_keyframes(&self, param: Params) -> Vec<(TimeType, f64)> {
                let $gkeyframes_s = &self.fields;
                match param {
                    $( Params::$f64_enum => { let $gkeyframes_p = &self.$f64_field; $gkeyframes_block }, )*
                    _ => panic!("Wrong parameter type"),
                }
            }
            fn set_f64_at_time(&mut self, param: Params, time: TimeType, value: f64) -> $crate::PluginResult<()> {
                let mut $stf64_s = &mut self.fields;
                match param {
                    $( Params::$f64_enum => { let $stf64_p = &mut self.$f64_field; let $stf64_t = time; let $stf64_v = value; $stf64_block }, )*
                    _ => panic!("Wrong parameter type"),
                }
            }
        }
    };
}
