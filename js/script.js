import * as tf from '@tensorflow/tfjs';
import * as tfjsconv from '@tensorflow/tfjs-converter';

import { Webcam } from './webcam';

console.log('hi');
var loc = window.location.pathname;
console.log(loc);
let model;
const MODEL_PATH = './webcam.py'
const WEIGHTS_PATH = '/web_model/weights_manifest.json'
const MODEL_SIZE = 224;
const TOPK_PREDICTIONS = 5;

const webcam = new Webcam(document.getElementById('webcam'));

(async function main() {
  try {
    alert("IMPORTANT: I do not store any images or video. Honestly I can't afford to do that. All inference is happening on-device and nothing is sent to me.");
    model = await tfjsconv.loadFrozenModel(MODEL_PATH, WEIGHTS_PATH);
    await webcam.setup();
    // model.predict(tf.zeros([1, IMAGE_SIZE, IMAGE_SIZE, 3])).dispose();

  } catch(e) {
    console.error(e);
  }
})();

document.getElementById('webcam').addEventListener('click', async () => {
  console.log('pic taken');
  const img = webcam.capture();
  // predict(img);
});

// async function predict(imgElement) {
//   status('Predicting...');

//   const startTime = performance.now();
//   const logits = tf.tidy(() => {
//     // tf.fromPixels() returns a Tensor from an image element.
//     const img = tf.fromPixels(imgElement).toFloat();

//     const offset = tf.scalar(127.5);
//     // Normalize the image from [0, 255] to [-1, 1].
//     const normalized = img.sub(offset).div(offset);

//     // Reshape to a single-element batch so we can pass it to predict.
//     const batched = normalized.reshape([1, IMAGE_SIZE, IMAGE_SIZE, 3]);

//     // Make a prediction through mobilenet.
//     return mobilenet.predict(batched);
//   });

//   // Convert logits to probabilities and class names.
//   const classes = await getTopKClasses(logits, TOPK_PREDICTIONS);
//   const totalTime = performance.now() - startTime;
//   status(`Done in ${Math.floor(totalTime)}ms`);

//   // Show the classes in the DOM.
//   showResults(imgElement, classes);
// }

// export async function getTopKClasses(logits, topK) {
//   const values = await logits.data();

//   const valuesAndIndices = [];
//   for (let i = 0; i < values.length; i++) {
//     valuesAndIndices.push({value: values[i], index: i});
//   }
//   valuesAndIndices.sort((a, b) => {
//     return b.value - a.value;
//   });
//   const topkValues = new Float32Array(topK);
//   const topkIndices = new Int32Array(topK);
//   for (let i = 0; i < topK; i++) {
//     topkValues[i] = valuesAndIndices[i].value;
//     topkIndices[i] = valuesAndIndices[i].index;
//   }

//   const topClassesAndProbs = [];
//   for (let i = 0; i < topkIndices.length; i++) {
//     topClassesAndProbs.push({
//       className: IMAGENET_CLASSES[topkIndices[i]],
//       probability: topkValues[i]
//     })
//   }
//   return topClassesAndProbs;
// }