export class CameraSensor {

    constructor(engine, scene, camera) {
        
        this.engine = engine
        this.scene = scene
        this.camera = camera
    }

    setupCameraSensor(obs_width, obs_height, posVector3) {  

        // Eine zweite Kamera (z. B. Sensor-Kamera) 
        const sensorCamera = new BABYLON.FreeCamera("sensorCam", posVector3 , this.scene);  // posVector3 = new BABYLON.Vector3(0, 11, 0)
        sensorCamera.setTarget(BABYLON.Vector3.Zero());

        // mit Box zur Visualisierung der Camera-Position
        const sensorCamBox = BABYLON.MeshBuilder.CreateBox("sensorCamBox", { width: 0.5, height: 0.5, depth: 0.25 }, this.scene);
        sensorCamBox.parent = sensorCamera
        sensorCamBox.position = new BABYLON.Vector3(0, 0, -0.05)

        // Material mit Wireframe für Camera Box
        const wireMat = new BABYLON.StandardMaterial("wireMat", this.scene);
        wireMat.wireframe = true;
        wireMat.emissiveColor = new BABYLON.Color3(0, 1, 0); // grün leuchtend
        sensorCamBox.material = wireMat;

        // Erstelle ein RenderTargetTexture
        this.renderTexture = new BABYLON.RenderTargetTexture("sensorRTT", { width: obs_width, height: obs_height }, this.scene, false, false);
        this.renderTexture.ignoreCameraViewport = true;
        this.renderTexture.renderList.push(...this.scene.meshes);
        this.renderTexture.activeCamera = sensorCamera;
        this.scene.customRenderTargets.push(this.renderTexture);

        // Erstelle eine Plane, die Texture von sensorcam zeigt
        this.viewPlane = BABYLON.MeshBuilder.CreatePlane("viewPlane", { width: 1, height: 1 }, this.scene);
        this.viewPlane.parent = this.camera
        this.viewPlane.position = new BABYLON.Vector3(5, -3, 19);  // tweak for size/offset
        this.viewPlane.rotation = BABYLON.Vector3.Zero();

        // Material mit Texture
        const rtMat = new BABYLON.StandardMaterial("rtMat", this.scene);
        rtMat.emissiveTexture = this.renderTexture;
        rtMat.disableLighting = true;            // ignore lights completely
        rtMat.backFaceCulling = false;           // optional: see it from both sides
        this.viewPlane.material = rtMat;

        // Pre-allocate pixel buffer 
        if (!this.pixelBuffer) {
            this.pixelBuffer = new Uint8Array(this.renderTexture.getSize().width * this.renderTexture.getSize().height * 4);
        }
    }

    async getVisualObs() {
        await this.renderTexture.readPixels(undefined, undefined, this.pixelBuffer);
        return this.pixelBuffer
    }

    // --- function to update HUD position on resize ---
    updateHudPosition() {
        const aspect = this.engine.getRenderWidth() / this.engine.getRenderHeight();
        const fov = this.camera.fov; // vertical FOV in radians

        // Distance from camera to plane
        const z = this.viewPlane.position.z;

        // compute visible height at this distance (in world units)
        const visibleHeight = 2 * z * Math.tan(fov / 2);
        const visibleWidth = visibleHeight * aspect;

        // now position plane near bottom-right corner
        const margin = 0.19; // how far from edges
        this.viewPlane.position.x = visibleWidth / 2 - this.viewPlane.scaling.x / 2 - margin;
        this.viewPlane.position.y = -visibleHeight / 2 + this.viewPlane.scaling.y / 2 + margin;
    }
}