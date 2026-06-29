import { Random, BaseEnv } from '../../../../lib/index.js'
import envConfig from './env-config.js';


export class BabylonEnv extends BaseEnv {

    constructor(wsUrl, canvasId) {
        super(wsUrl, canvasId, envConfig);

        // Variables important for training
        this.options = {
            size: 30.0,
            speed: 20.0,
            dt: 0.02,
            reward: 5.0,
            punishment: -5.0
        }

        this.fixedDt = this.options.dt
        this.delta = this.options.speed * this.options.dt;

        this.setupAll();
    }

    
    async setupScene() {

        // Setup the RL-Scene 
        this.camera = new BABYLON.ArcRotateCamera("camera", Math.PI / 4, Math.PI / 4, this.options.size * 2, BABYLON.Vector3.Zero(), this.scene);
        this.camera.attachControl(this.canvas, true);

        const environment = this.scene.createDefaultEnvironment({
            createGround: false,
            createSkybox: false
        });

        this.scene.environmentIntensity = 1.0;

        // Load model from Blender
        await BABYLON.SceneLoader.ImportMeshAsync("", "./js/models/", "3dmodel.glb", this.scene)

        this.agent = this.scene.getMeshById("agent")
        this.target = this.scene.getMeshById("target")
    }

    async setupUI() {

        // Setup UI - actual delta variabl is saved in --> this.UI.DELTA
        this.UI.addSlider("Delta", 0.1, 1, 0.1, this.delta)
    }


    async resetEnv() {

        const bounds = this.options.size / 2

        // Reset RL-Scene
        this.agent.position = new BABYLON.Vector3(0, 0.5, 0);
        this.target.position = new BABYLON.Vector3(Random.randomBetween(-bounds, bounds), 0.5, Random.randomBetween(-bounds, bounds));

        // Render scene
        this.scene.render()

        const s = this.result.data.state;

        s[0] = this.agent.position.x;
        s[1] = this.agent.position.z;
        s[2] = this.target.position.x;
        s[3] = this.target.position.z;

        // Return state
        return this.result
    }

    
    async stepEnv(action) {

        switch (action) {
            case 0: this.agent.position.z += this.UI.DELTA; break;  // forward
            case 1: this.agent.position.z -= this.UI.DELTA; break;  // backward
            case 2: this.agent.position.x -= this.UI.DELTA; break;  // right
            case 3: this.agent.position.x += this.UI.DELTA; break;  // left
        }

        // compute distance target - agent
        const distance = BABYLON.Vector3.Distance(this.agent.position, this.target.position);

        // higher reward means lower distance -3, -1, -0.1 is higher reward because of the minus sign
        let reward = 0.2 * (1 / (1 + distance));

        // check for collision 
        let terminated = distance < 1.0;

        // if reached goal give extrta bonus reward
        if (terminated) {
            reward += this.options.reward;
        };

        const bounds = this.options.size / 2

        // if out of bound terminate and negative reward
        if (this.agent.position.z <= -bounds || this.agent.position.z >= bounds || this.agent.position.x <= -bounds || this.agent.position.x >= bounds) {
            terminated = true
            reward = this.options.punishment
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

        data.reward = reward;
        data.terminated = terminated;
        data.truncated = truncated;

        // Return state, reward, done [truncated or terminated]
        return this.result
    }

    async curriculumLevel(level) {
    }
}

