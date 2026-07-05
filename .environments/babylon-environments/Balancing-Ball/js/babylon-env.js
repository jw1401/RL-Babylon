// BabylonEnv.js implements the RL Environment for Training.

import { BaseEnv } from "../../../lib/index.js";
import envConfig from './env-config.js';

export class BabylonEnv extends BaseEnv {

    constructor(wsUrl, canvasId) {
        super(wsUrl, canvasId, envConfig);

        this.options = {
            gravity: -9.81,
            dt: 0.02,                                       // 50 HZ framerate
            platformSize: { w: 4, d: 4, h: 0.25 },
            ballRadius: 0.2,
            maxTilt: 0.35,
            tiltStep: 0.01,
            fallY: -1,
            rewardAlive: 0.1,
            rewardFall: -5.0
        }

        this.fixedDt = this.options.dt
        this.setupAll();
    }

    // Setup the RL Scene 
    async setupScene() {

        // Camera + Light
        const cam = new BABYLON.ArcRotateCamera("camera", -Math.PI / 2, Math.PI / 3, 8, new BABYLON.Vector3(0, 1, 0), this.scene);
        cam.attachControl(this.canvas, true);
        new BABYLON.HemisphericLight("light", new BABYLON.Vector3(0, 1, 0), this.scene);

        // ---- INIT HAVOK PHYSICS ----
        const havok = await HavokPhysics();
        const plugin = new BABYLON.HavokPlugin(false, havok);
        this.scene.enablePhysics(new BABYLON.Vector3(0, this.options.gravity, 0), plugin);

        this._createPlatform();
        this._createBall();

        this.px = 0;
        this.pz = 0;

        const ph = this.scene.getPhysicsEngine()
        ph.setTimeStep(this.options.dt)
    }

    // Setup individual UI
    setupUI() {
    }

    // Reset logic for RL Scene
    async resetEnv() {

        this.px = 0
        this.pz = 0

        this._applyPlatformRotationPhysics()

        this.ball.dispose();
        this._createBall();

        // Render scene
        this.scene.render()
        this._setObs()

        // Return state
        return this.result
    }

    // Step logic for RL Scene
    async stepEnv(action) {

        this._applyAction(action);

        let reward = this.options.rewardAlive;
        let terminated = false

        const distance = BABYLON.Vector3.Distance(this.platform.position, this.ball.position)
        const distanceReward = (1 / distance) * 0.1

        reward += distanceReward
        // console.log(reward)

        if (this.ball.position.y < this.options.fallY) {
            reward = this.options.rewardFall;
            terminated = true;
        }

        // if max steps reached then episode is truncated == Time Horizon
        const truncated = this.stepCount >= this.UI.MAX_STEPS;

        // Render Scene
        this.scene.render()
        this._setObs()

        const data = this.result.data;
        data.reward = reward;
        data.terminated = terminated;
        data.truncated = truncated;

        // Return state, reward, done [truncated or terminated]
        return this.result
    }

    _setObs() {

        const s = this.result.data.state;
        const pos = this.ball.position;
        const vel = this.ballAggregate.body.getLinearVelocity();

        s[0] = pos.x;
        s[1] = pos.y;
        s[2] = pos.z;
        s[3] = vel.x;
        s[4] = vel.y;
        s[5] = vel.z;
        s[6] = this.platform.rotation.x;
        s[7] = this.platform.rotation.z;
    }

    _createPlatform() {

        const o = this.options;

        this.platform = BABYLON.MeshBuilder.CreateBox("platform", { width: o.platformSize.w, depth: o.platformSize.d, height: o.platformSize.h, }, this.scene);
        this.platform.position.y = 0;

        // Visual material
        const mat = new BABYLON.StandardMaterial("pm", this.scene);
        mat.diffuseColor = new BABYLON.Color3(0.7, 0.7, 1.0);
        this.platform.material = mat;

        // Physics V2 — static body
        this.platformAggregate = new BABYLON.PhysicsAggregate(this.platform, BABYLON.PhysicsShapeType.BOX, { mass: 0 }, this.scene);
    }

    _createBall() {

        const o = this.options;

        this.ball = BABYLON.MeshBuilder.CreateSphere("ball", { diameter: o.ballRadius * 2 }, this.scene);
        this.ball.position = new BABYLON.Vector3(0, 1.2, 0);

        // Visual
        const mat = new BABYLON.StandardMaterial("bm", this.scene);
        mat.diffuseColor = new BABYLON.Color3(1, 0.3, 0.3);
        this.ball.material = mat;

        // Physics body (dynamic)
        this.ballAggregate = new BABYLON.PhysicsAggregate(this.ball, BABYLON.PhysicsShapeType.SPHERE, { mass: 1, restitution: 0.05, friction: 0.2 }, this.scene);
    }

    _applyPlatformRotationPhysics() {

        // 1. Rotate the mesh normally
        this.platform.rotationQuaternion = BABYLON.Quaternion.FromEulerAngles(this.px, 0, this.pz);

        // 2. Remove old collider
        this.platformAggregate.dispose();

        // 3. Create new collider at the new orientation
        this.platformAggregate = new BABYLON.PhysicsAggregate(this.platform, BABYLON.PhysicsShapeType.BOX, { mass: 0 }, this.scene);
    }

    _applyAction(action) {

        const s = this.options.tiltStep;

        switch (action) {
            case 0: this.px += s; break;  // forward
            case 1: this.px -= s; break;  // backward
            case 2: this.pz += s; break;  // right
            case 3: this.pz -= s; break;  // left
        }

        const max = this.options.maxTilt;
        this.px = Math.max(-max, Math.min(max, this.px));
        this.pz = Math.max(-max, Math.min(max, this.pz));

        // IMPORTANT: update physics transform
        this._applyPlatformRotationPhysics();
    }
}

