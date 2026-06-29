// ============================================================
//  UIController.js
// ------------------------------------------------------------
//  Implements:
//    • Adding slides controlls programmatically --> prop values 
//      can be acessed from outside with name in uppercase --> 
//      this.UI.delta, ....
//    • Set details UI
//    • Set fps display
// ============================================================

export class UIController {

    constructor(containerId, envConfig) {

        this.envConfig = envConfig

        this.fpsLabel = document.getElementById("fpsLabel");
        this.container = document.getElementById(containerId);

        // Setup UI 
        this._setupDetailsUI()
    }

    addSlider(name, min, max, step, initialValue) {
        const prop = name.toUpperCase();        // e.g. "Delta" → "DELTA"
        const labelId = `${prop}Label`;
        const sliderId = `${prop}Slider`;

        // initialize the property dynamically
        this[prop] = initialValue;

        // build HTML
        const html = `
          <label class="ui_label" title="${prop}">${name.replace(/_/g, ' ')}: <span id="${labelId}">${initialValue}</span></label><br>
          <input class="ui_slider" type="range" id="${sliderId}" min="${min}" max="${max}" step="${step}" value="${initialValue}">
          <br>
        `;

        // use insertAdjacentHTML instead of innerHTML +=
        this.container.insertAdjacentHTML('beforeend', html);

        const slider = this.container.querySelector(`#${sliderId}`);
        const label = this.container.querySelector(`#${labelId}`);

        // keep the property updated
        slider.addEventListener("input", () => {
            const val = parseFloat(slider.value);
            this[prop] = val;                 // 👈 updates `this.delta`, `this.gamma`, etc.
            label.textContent = val;
        });
    }


    setFPS(fps) {
        this.fpsLabel.innerText = fps;
    }


    // Setup Details UI on init 
    _setupDetailsUI() {
        const container = document.getElementById("details");                    // parent element
        container.innerHTML = "<summary>Details</summary><hr>"

        // console.log("\n === Environment Description === \n\n")

        for (const [key, value] of Object.entries(this.envConfig)) {

           // console.log(`${key.toUpperCase()}: ${value}`)

            const p = document.createElement("p");                            // create a <p> for each pair
            p.innerHTML = `<strong>${key.toUpperCase()}:</strong> ${value}`;  // add text
            container.appendChild(p);                                         // add <p> to the container
        }
        // console.log("\n\n\n")

        document.body.appendChild(container);
    }





}
