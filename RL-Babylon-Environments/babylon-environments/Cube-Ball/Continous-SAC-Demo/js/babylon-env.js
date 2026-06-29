import { Random, BaseEnv } from '../../../../lib/index.js'
import envConfig from './env-config.js';

// BabylonEnv.js implements the RL Environment for Training. It and is based on BaseEnv 
//
export class BabylonEnv extends BaseEnv {

    constructor(wsUrl, canvasId) {

        super(wsUrl, canvasId, envConfig);

        // Variables important for training
        this.size = 10.0;
        this.fixedDt = 0.02                         // tick time = 20 ms = 0.02 sec --> 50 HZ framerate 
        this.velocity = { x: 0, z: 0 };
        this.friction = 2.0
        this.accelerationScale = 100.0
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
        this.ground = BABYLON.MeshBuilder.CreateGround("ground", { width: this.size, height: this.size, subdivisions: this.size }, this.scene);
    }

    async setupUI() {
    }


    async resetEnv() {

        // Reset logic for RL Scene
        this.agent.position = new BABYLON.Vector3(0, 0.5, 0);
        this.target.position = new BABYLON.Vector3(Random.randomBetween(-this.size / 2, this.size / 2), 0.5, Random.randomBetween(-this.size / 2, this.size / 2));

        this.velocity.x = 0
        this.velocity.z = 0

        // Render scene
        this.scene.render()

        const s = this.result.data.state;
        s[0] = this.agent.position.x;
        s[1] = this.agent.position.z;
        s[2] = this.target.position.x;
        s[3] = this.target.position.z;
        s[4] = this.velocity.x;
        s[5] = this.velocity.z;

        // Return state
        return this.result
    }

    // Step logic for RL Scene
    async stepEnv(action) {

        // 1. Convert incoming actions into v
        this.velocity.z += action.z * this.accelerationScale * this.fixedDt;
        this.velocity.x += action.x * this.accelerationScale * this.fixedDt;

        // 2. Apply friction (Damping)
        // This reduces the velocity by a percentage every frame
        this.velocity.z *= Math.exp(-this.friction * this.fixedDt);
        this.velocity.x *= Math.exp(-this.friction * this.fixedDt);

        // 3. Apply velocity to position
        this.agent.position.z += this.velocity.z * this.fixedDt;
        this.agent.position.x += this.velocity.x * this.fixedDt;

        // compute distance target - agent
        const distance = BABYLON.Vector3.Distance(this.agent.position, this.target.position);

        // higher reward means lower distance -3, -1, -0.1 is higher reward because of the minus sign
        let reward = -0.01 //0.01 * (1 / (1 + distance));

        // check for collision 
        let terminated = distance < 1.0;

        // if reached goal give extrta bonus reward
        if (terminated) {
            reward += 10.0;
            // console.log("yeah")
        };

        // if out of bound terminate and negative reward
        if (this.agent.position.z <= -this.size / 2 || this.agent.position.z >= this.size / 2 || this.agent.position.x <= -this.size / 2 || this.agent.position.x >= this.size / 2) {
            terminated = true
            reward = -5.0
        }

        // if max steps reached then episode is truncated == Time Horizon
        const truncated = this.stepCount >= this.UI.MAX_STEPS;

        // Render Scene
        this.scene.render()

        const data = this.result.data;
        const s = data.state;
        s[0] = this.agent.position.x;
        s[1] = this.agent.position.z;
        s[2] = this.target.position.x;
        s[3] = this.target.position.z;
        s[4] = this.velocity.x;
        s[5] = this.velocity.z;

        data.reward = reward;
        data.terminated = terminated;
        data.truncated = truncated;

        // Return state, reward, done [truncated or terminated]
        return this.result
    }

    async curriculumLevel(level) {
    }
}

