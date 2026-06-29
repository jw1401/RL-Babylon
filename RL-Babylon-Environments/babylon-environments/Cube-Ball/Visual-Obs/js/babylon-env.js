import { CameraSensor, BaseEnv, Random } from '../../../../lib/index.js'
import envConfig from './env-config.js';

// BabylonEnv.js implements the RL Environment for Training. It and is based on BaseEnv 
//
export class BabylonEnv extends BaseEnv {

    constructor(wsUrl, canvasId) {

        super(wsUrl, canvasId, envConfig);

        // Variables important for training
        this.size = 10.0;
        const speed = 10                            // 10 m/s = 36 km/h
        this.fixedDt = 0.02                         // tick time = 20 ms = 0.02 sec --> 50 HZ framerate 
        this.delta = speed * this.fixedDt;          // Compute delta for animation update based on speed

        this.setupAll();
    }

    // Setup the RL Scene based on individula design
    async setupScene() {

        // Camera and light
        this.camera = new BABYLON.ArcRotateCamera("camera", Math.PI / 4, Math.PI / 4, 65, BABYLON.Vector3.Zero(), this.scene);
        this.camera.attachControl(this.canvas, true);
        new BABYLON.HemisphericLight("light", new BABYLON.Vector3(1, 1, 0), this.scene);

        // Agent
        this.agent = BABYLON.MeshBuilder.CreateBox("agent", { size: 1 }, this.scene);
        this.agent.position.y = 0.5;
        this.agent.position.z = 3.0

        const agentMat = new BABYLON.StandardMaterial("agentMat", this.scene);
        agentMat.diffuseColor = new BABYLON.Color3(0, 0, 1); // green
        agentMat.emissiveColor = agentMat.diffuseColor; // make it bright, unaffected by lighting
        this.agent.material = agentMat;

        // Target
        this.target = BABYLON.MeshBuilder.CreateSphere("target", { diameter: 1 }, this.scene);
        this.target.position.y = 0.5;
        const targetMaterial = new BABYLON.StandardMaterial("targetMat", this.scene);
        targetMaterial.diffuseColor = new BABYLON.Color3(1, 0, 0); // red color
        this.target.material = targetMaterial;

        // Create a ground with size 20 x 20 and grid subdivision 20 
        const ground = BABYLON.MeshBuilder.CreateGround("ground", { width: this.size, height: this.size, subdivisions: this.size }, this.scene);

        this.secondCamera = new CameraSensor(this.engine, this.scene, this.camera)
        this.secondCamera.setupCameraSensor(this.state_space[0], this.state_space[1], new BABYLON.Vector3(0, 12, 0))
    }

    // Setup individual UI
    setupUI() {
        this.UI.addSlider("Delta", 0.1, 1, 0.1, this.delta)
    }

    // Reset logic for RL Scene
    async resetEnv() {

        this.agent.position = new BABYLON.Vector3(0, 0.5, 0);

        let rp = (this.size / 2) - 0.5
        this.target.position = new BABYLON.Vector3(Random.randomBetween(-rp, rp), 0.5, Random.randomBetween(-rp, rp));

        // Render before get pixels to get actual state and return state as visual 84x84 RGBA 4 channel frame
        this.scene.render()

        const obs = await this.secondCamera.getVisualObs();
        this.result.frame = obs

        return this.result
    }

    // Step logic for RL Scene
    async stepEnv(action) {

        // Steering with actions send by python trainer
        if (action === 0) this.agent.position.z += this.UI.DELTA;
        if (action === 1) this.agent.position.z -= this.UI.DELTA;
        if (action === 2) this.agent.position.x -= this.UI.DELTA;
        if (action === 3) this.agent.position.x += this.UI.DELTA;

        // compute distance target - agent
        const distance = BABYLON.Vector3.Distance(this.agent.position, this.target.position);

        // higher reward means lower distance -3, -1, -0.1 is higher reward because of the minus sign
        let reward = 0.2 * (1 / (1 + distance));

        // check for collision 
        let terminated = distance < 1.0;

        // if reached goal give extrta bonus reward
        if (terminated) {
            reward += 5.0;
            console.log("yeah")
        };

        // if out of bound terminate and negative reward
        if (this.agent.position.z <= -this.size / 2 || this.agent.position.z >= this.size / 2 || this.agent.position.x <= -this.size / 2 || this.agent.position.x >= this.size / 2) {
            terminated = true
            reward = -5.0
        }

        // if max steps reached then episode is truncated == Time Horizon
        const truncated = this.stepCount >= this.UI.MAX_STEPS;

        this.scene.render()

        const obs = await this.secondCamera.getVisualObs();
        const data = this.result.data
        this.result.frame = obs
        data.reward = reward;
        data.terminated = terminated;
        data.truncated = truncated;

        // return visual state, reward, done [truncated or terminated]
        return this.result
    }
}

