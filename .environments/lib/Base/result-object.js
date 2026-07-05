export class ResultObject {
    constructor(stateSpace) {
        this.data = {
            state: new Array(stateSpace).fill(0.0),
            reward: 0.0,
            terminated: false,
            truncated: false
        };
        this.frame = null;
    }
}