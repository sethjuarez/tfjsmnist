// Vue component for drawing and
// converting hand-drawn digits
// to predicted numbers
Vue.component('drawing-board', {
    data: function () {
        return {
            mouse: {
                current: {
                    x: 0,
                    y: 0
                },
                down: false
            },
            model: null,
            probabilities: [],
            prediction: -1
        }
    },
    computed: {
        currentMouse: function () {
            var c = document.getElementById("canvas");
            var rect = c.getBoundingClientRect();

            return {
                x: this.mouse.current.x - rect.left,
                y: this.mouse.current.y - rect.top
            }
        }
    },
    template: `
        <div id="ml">
            <div id="draw">
                <canvas id="canvas" 
                        v-on:mousedown="handleMouseDown" 
                        v-on:mouseup="handleMouseUp" 
                        v-on:mousemove="handleMouseMove" 
                        width="250px" 
                        height="250px">
                </canvas>
                <canvas id="mini" width="28" height="28"></canvas>
            </div>
            <div id="predict" v-if="prediction > -1">
                <div id="prediction">I think it's a {{prediction}}</div>
                <ol start="0">
                    <li v-for="p in probabilities">{{p.toPrecision(3)}}</li>
                </ol>
            </div>
            <div id="buttons">
                <button v-on:click="reset">Reset</button>
                <button v-on:click="predict">Predict</button>
            </div>
        </div>
    `,
    methods: {
        draw: function (event) {
            if (this.mouse.down) {
                var c = document.getElementById("canvas");
                var ctx = c.getContext("2d");
                ctx.lineTo(this.currentMouse.x, this.currentMouse.y);
                ctx.strokeStyle ="#00000";
                ctx.lineWidth = 15;
                ctx.stroke()
            }
        },
        clear: function (el) {
            var c = document.getElementById(el);
            var ctx = c.getContext("2d");
            ctx.beginPath();
            ctx.clearRect(0,0,c.width,c.height);
            ctx.closePath();
        },
        handleMouseDown: function (event) {
            this.mouse.down = true;

            this.mouse.current = {
                x: event.pageX,
                y: event.pageY
            }
            var c = document.getElementById("canvas");
            var ctx = c.getContext("2d");
            ctx.moveTo(this.currentMouse.x, this.currentMouse.y)
        },
        handleMouseUp: function () {
            this.mouse.down = false;
        },
        handleMouseMove: function (event) {
            this.mouse.current = {
                x: event.pageX,
                y: event.pageY
            }
            this.draw(event)
        },
        reset: function(event) {
            this.clear("canvas");
            this.clear("mini");
            this.prediction = -1;
        },
        getDigit: function() {
            var c = document.getElementById("canvas");
            // redraw digit to appropriate size
            var t = document.getElementById("mini");
            var ctx = t.getContext("2d");
            // clear previous image
            ctx.beginPath();
            ctx.clearRect(0, 0, t.width, t.height);
            ctx.closePath();

            // resize
            ctx.drawImage(c, 0, 0, t.width, t.height);

            // get image pixels
            var image = ctx.getImageData(0, 0, t.width, t.height);
            var data = image.data;

            // digit tensor
            var digit = [];

            for(var i = 0; i < data.length; i+=4) {
                // for some reason the alpha
                // channel holds the right data
                var v = data[i + 3];

                // add+normalize
                digit.push(v / 255.);
                
                // we will redraw anyway as a
                // good sanity check
                data[i] =      v;
                data[i + 1] =  v;
                data[i + 2] =  v;
                data[i + 3] = 255;
            }
            ctx.putImageData(image, 0, 0);

            return digit;
        },
        predict: async function() {
            const digit = this.getDigit();
            const d = tf.tensor1d(digit).reshape([1, digit.length]);
            this.probabilities = await this.model.execute({'x': d}).reshape([10]).data();
            this.prediction = this.probabilities.indexOf(Math.max(...this.probabilities));
        }
    },
    mounted: async function () {
        var c = document.getElementById("canvas");
        var ctx = c.getContext("2d");
        ctx.translate(0.5, 0.5);
        ctx.imageSmoothingEnabled= true;

        // load model
        const MODEL = 'model/tensorflowjs_model.pb';
        const WEIGHTS = 'model/weights_manifest.json';
        this.model = await tf.loadFrozenModel(MODEL, WEIGHTS);
    }
})

var app = new Vue({ 
    el: '#app',
});