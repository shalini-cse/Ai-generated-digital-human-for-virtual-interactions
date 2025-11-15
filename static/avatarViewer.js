let scene, camera, renderer, avatar;

init();
animate();

function init() {
  scene = new THREE.Scene();
  camera = new THREE.PerspectiveCamera(45, window.innerWidth/window.innerHeight, 0.1, 100);
  camera.position.set(0, 1.5, 3);

  renderer = new THREE.WebGLRenderer({antialias: true});
  renderer.setSize(window.innerWidth, window.innerHeight);
  document.body.appendChild(renderer.domElement);

  const light = new THREE.DirectionalLight(0xffffff, 1);
  light.position.set(2, 5, 5);
  scene.add(light);

  const loader = new THREE.GLTFLoader();
  loader.load("https://models.readyplayer.me/684aa1aa2f4b144e199f50a9.glb", function(gltf) {
    avatar = gltf.scene;
    avatar.position.set(0, -1.5, 0);
    scene.add(avatar);
  });
}

function animate() {
  requestAnimationFrame(animate);
  renderer.render(scene, camera);
}

function sendPrompt() {
  const input = document.getElementById("userInput").value;
  fetch("/reply", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message: input })
  })
  .then(res => res.json())
  .then(data => {
    document.getElementById("aiResponse").innerText = "AI: " + data.reply;
    const audio = new Audio("/static/output.mp3");
    audio.play();
    // TODO: lip sync and emotion trigger here
  });
}
// Duplicate loader declarations removed to fix redeclaration error
// If you need to load another model, use the existing loader or create a new uniquely named variable

camera = new THREE.PerspectiveCamera(45, window.innerWidth/window.innerHeight, 0.1, 100);
camera.position.set(0, 1.4, 2.5);  // move closer or higher

const light = new THREE.DirectionalLight(0xffffff, 2);
light.position.set(0, 4, 5);
scene.add(light);
