// this is the main prediction function!
var predictMNIST = function(digits) {
    console.log(digits);
};

// Vue component for drawing and
// converting hand-drawn digits
Vue.component('drawing-board', {
    data: function () {
        return {
            mouse: {
                current: {
                    x: 0,
                    y: 0
                },
                previous: {
                    x: 0,
                    y: 0
                },
                down: false
            }
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
        <div>
            <canvas id="canvas" 
                    v-on:mousedown="handleMouseDown" 
                    v-on:mouseup="handleMouseUp" 
                    v-on:mousemove="handleMouseMove" 
                    width="200px" 
                    height="200px">
            </canvas>
            <canvas id="mini" width="28" height="28"></canvas>
            <div>
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
        },
        predict: function() {
            var c = document.getElementById("canvas");
            // redraw digit to appropriate size
            var t = document.getElementById("mini");
            var ctx = t.getContext("2d");
            // clear previous image
            ctx.beginPath();
            ctx.clearRect(0,0,t.width,t.height);
            ctx.closePath();

            // resize
            ctx.drawImage(c, 0, 0, t.width, t.height);

            // get image pixels
            var image = ctx.getImageData(0, 0, t.width, t.height);
            var data = image.data;

            // digit data
            var digit = new Uint8ClampedArray(t.width * t.height);
            for(var i = 0; i < data.length; i+=4) {
                // for some reason the alpha
                // channel holds the right data
                // we will redraw anyway as a
                // good sanity check
                digit[i % 4] = data[i + 3];
                data[i] = data[i + 3];
                data[i + 1] = data[i + 3];
                data[i + 2] = data[i + 3];
                data[i + 3] = 255;
            }
            ctx.putImageData(image, 0, 0);
            predictMNIST(digit);
        }
    },
    ready: function () {
        var c = document.getElementById("canvas");
        var ctx = c.getContext("2d");
        ctx.translate(0.5, 0.5);
        ctx.imageSmoothingEnabled= true;
    }
})

var app = new Vue({ 
    el: '#app',
});