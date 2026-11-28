# Where does the point land in the image?

A camera tilted down by an angle θ, one marked point P, and the picture the camera takes. Tilt the camera and
watch P move in the image. It uses the simple image formation model from Module 3, Lecture 1:
x = X + x₀ and y = cos(θ) Y − sin(θ) Z + y₀, with x₀ = y₀ = 0.

Not yet in a course. It is planned as a Module 3, Lecture 1 interactive, after the segment that teaches the model.

## Open in your browser

- [3D version](https://codey-m.github.io/cv_nlp/demos/camera_tilt/widget-camera-tilt-3d.html): turn the scene by
  dragging, or with the arrow keys once it has focus.
- [2D version](https://codey-m.github.io/cv_nlp/demos/camera_tilt/widget-camera-tilt.html): the same model drawn
  from the side.

The two are kept so they can be compared. The 3D version loads Three.js from this site's shared copy in
`widgets/lib/three-r186/`, so review it from its link: a downloaded copy cannot load it. The 2D version is one
self-contained file and also works when downloaded.
