import { BsonSocketClient } from './bson-socket-client.js';
import { UIController } from '../UI/ui-controller.js';
import { Performance } from '../Utils/performance.js';
import { ResultObject } from './result-object.js';

export class BaseEnv {

    constructor(wsUrl, canvasId, envConfig) {

        // Websocket setup URL and websocket itself
        this._wsUrl = wsUrl;

        // Variables important for training
        this.envConfig = envConfig;
        this.state_space = envConfig.state_space;
        this.action_space = envConfig.action_space;
        this.vis_obs = envConfig.VIS_OBS;

        // MAX_STEPS is Time horizon - number of steps till truncation
        this.MAX_STEPS = envConfig.MAX_STEPS;
        this.stepCount = 0;

        // Setup Render-Engine
        this.canvas = document.getElementById(canvasId);
        this.engine = new BABYLON.Engine(this.canvas, true);
        this.scene = new BABYLON.Scene(this.engine);

        this.realtime = false;
        this.fixedDt = 0; // defined in BabylonEnv 

        this.UI = new UIController("ui", this.envConfig);
        this.Performance = new Performance();
        this.result = new ResultObject(this.vis_obs ? 0 : this.state_space)

        // Netzwerk-Modul instanziieren & Events verknüpfen
        this.socket = new BsonSocketClient(this._wsUrl);
        this._setupNetworkEvents();

    }

    _setupNetworkEvents() {

        // Hier binden wir die Netzwerk-Ereignisse an unsere internen Handler
        this.socket.on('init', () => this._handleInit());
        this.socket.on('reset', () => this._handleReset());
        this.socket.on('action', (action) => this._handleStep(action));
        this.socket.on('curriculum_level', (level) => this._handleCurriculumLevel(level));
    }

    async setupAll() {

        // Setup scene, UI and connect to RL trainer
        await this.setupScene();
        this.scene.executeWhenReady(() => { this.scene.render(); });

        // Set max steps for truncation = Time horizon in RL
        this.UI.addSlider("Max_Steps", 10, this.MAX_STEPS, 10, this.MAX_STEPS);
        await this.setupUI();

        // Start connection
        this.socket.connect();
    }

    _handleInit() {

        this.socket.send({
            VIS_OBS: this.vis_obs,
            state_space: this.state_space,
            action_space: this.action_space
        });
    }

    async _handleReset() {

        this.stepCount = 0;
        const result = await this.resetEnv();
        
        this.socket.send({
            data: result.data,
            frame: result.frame 
        });
    }

    async _handleStep(action) {
        
        this.stepCount++;
        this.UI.setFPS(this.Performance.computeFPSWithSamples());

        const result = await this.stepEnv(action);

        if (this.realtime) {
            await this.Performance.sleep(this.fixedDt * 1000);
        }

        this.socket.send({
            data: result.data,
            frame: result.frame 
        });
    }

    async _handleCurriculumLevel(level) {

        // console.log("Received curriculum level: ", level);
        // Hier kannst du die Logik implementieren, um das Curriculum Level in deiner Umgebung zu berücksichtigen
        // Zum Beispiel könntest du die Schwierigkeit der Umgebung basierend auf dem Level anpassen

        await this.curriculumLevel(level);
        this.socket.send({ curriculum_level: level }); // Bestätige den Empfang des Curriculum Levels
    }

    async setupScene() {
        throw new Error("setupScene() must be implemented by subclass");
    }

    async setupUI() {
        throw new Error("setupUI() must be implemented by subclass");
    }

    async resetEnv() {
        throw new Error("resetEnv() must be implemented by subclass");
    }

    async stepEnv(action) {
        throw new Error("stepEnv() must be implemented by subclass");
    }

    async curriculumLevel(level) {
        console.log("no curriculum level handler implemented, received level: ", level);
        //throw new Error("curriculumLevel() must be implemented by subclass");
    }

}
