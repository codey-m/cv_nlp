three.js r186 (three@0.186.1, MIT, see LICENSE): the shared, pinned copy every widget that needs Three.js loads (WP-18).
Course widgets import it as ./lib/three-r186/...; demos in the labs repo as ../../widgets/lib/three-r186/....
docs/publish_widgets.py copies this folder to the labs repo's widgets/lib/ whenever a published widget or a demo uses it. Never edit these files; to upgrade, add a new versioned folder
(lib/three-r<N>/), move widgets over one at a time and rerun each one's checks.
Used by: CV_NLP_Labs/demos/camera_tilt/widget-camera-tilt-3d.html.

three.module.min.js and three.core.js are NOT files the three.js project publishes. The npm package's build/ folder ships
only unminified three.module.js and three.core.js. jsDelivr generated these two on request (2026-11-27) by running
Terser v5.48.0 over those official files; each file's header says so and names the original.

three.core.js keeps that name because three.module.min.js imports it by that name; its contents are minified.
OrbitControls.js is examples/jsm/controls/OrbitControls.js from the same package, unmodified and unminified.

For a reproducible copy, minify the official build/three.module.js and build/three.core.js locally (Terser or esbuild)
and replace these two files. Then rerun the checks of every widget that uses this folder.
