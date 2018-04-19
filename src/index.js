import * as tf from '@tensorflow/tfjs';
import * as tfc from '@tensorflow/tfjs-core';
import {loadFrozenModel, FrozenModel} from '@tensorflow/tfjs-converter';

/**
 * A class that wraps webcam video elements to capture Tensor4Ds.
 */
export class Webcam {
  /**
   * @param {HTMLVideoElement} webcamElement A HTMLVideoElement representing the webcam feed.
   */
  constructor(webcamElement) {
    this.webcamElement = webcamElement;
  }

  /**
   * Captures a frame from the webcam and normalizes it between -1 and 1.
   * Returns a batched image (1-element batch) of shape [1, w, h, c].
   */
  capture() {
    return tf.tidy(() => {
      // Reads the image as a Tensor from the webcam <video> element.
      const webcamImage = tf.image.resizeBilinear(tf.fromPixels(this.webcamElement, 3), [224, 224]);
      // console.log(webcamImage.data());

      // Crop the image so we're using the center square of the rectangular
      // webcam.
      const croppedImage = this.cropImage(webcamImage);

      // Expand the outer most dimension so we have a batch size of 1.
      const batchedImage = croppedImage.expandDims(0);

      // Normalize the image between -1 and 1. The image comes in between 0-255,
      // so we divide by 127 and subtract 1
      return croppedImage;
      // return croppedImage.toFloat().div(tf.scalar(255));
    });
    // return tf.fromPixels(this.webcamElement);
  }

  /**
   * Crops an image tensor so we get a square image with no white space.
   * @param {Tensor4D} img An input image Tensor to crop.
   */
  cropImage(img) {
    const size = Math.min(img.shape[0], img.shape[1]);
    // const size = 224;
    const centerHeight = img.shape[0] / 2;
    const beginHeight = centerHeight - (size / 2);
    const centerWidth = img.shape[1] / 2;
    const beginWidth = centerWidth - (size / 2);
    return img.slice([beginHeight, beginWidth, 0], [size, size, 3]);
  }

  /**
   * Adjusts the video size so we can make a centered square crop without
   * including whitespace.
   * @param {number} width The real width of the video element.
   * @param {number} height The real height of the video element.
   */
  adjustVideoSize(width, height) {
    const aspectRatio = width / height;
    if (width >= height) {
      this.webcamElement.width = aspectRatio * this.webcamElement.height;
    } else if (width < height) {
      this.webcamElement.height = this.webcamElement.width / aspectRatio;
    }
    // this.webcamElement.width = 224;
    // this.webcamElement.height = 224;
  }

  async setup() {
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
      const stream = await navigator.mediaDevices.getUserMedia({
        'audio': false,
        'video': {
          facingMode: 'user'
        }
      });
      window.stream = stream;
      this.webcamElement.srcObject = stream;
      return new Promise(resolve => {
        this.webcamElement.onloadedmetadata = () => {
          this.adjustVideoSize(
            this.webcamElement.videoWidth,
            this.webcamElement.videoHeight);
          resolve();
        };
      });
    } else {
      throw new Error('No webcam found!');
    }
  }
}

const classes = ['acne vulgaris', 'acrokeratosis verruciformis', 'actinic solar damage actinic cheilitis ', 'actinic solar damage actinic keratosis ', 'actinic solar damage cutis rhomboidalis nuchae ', 'actinic solar damage pigmentation ', 'actinic solar damage solar elastosis ', 'actinic solar damage solar purpura ', 'actinic solar damage telangiectasia ', 'acute eczema', 'allergic contact dermatitis', 'alopecia areata', 'androgenetic alopecia', 'angioma', 'angular cheilitis', 'aphthous ulcer', 'apocrine hydrocystoma', 'arsenical keratosis', 'balanitis xerotica obliterans', 'basal cell carcinoma', 'beau s lines', 'becker s nevus', 'behcet s syndrome', 'benign keratosis', 'blue nevus', 'bowen s disease', 'bowenoid papulosis', 'cafe au lait macule', 'callus', 'candidiasis', 'cellulitis', 'chalazion', 'clubbing of fingers', 'compound nevus', 'congenital nevus', 'crowe s sign', 'cutanea larva migrans', 'cutaneous horn', 'cutaneous t cell lymphoma', 'cutis marmorata', 'darier white disease', 'dermatofibroma', 'dermatosis papulosa nigra', 'desquamation', 'digital fibroma', 'dilated pore of winer', 'discoid lupus erythematosus', 'disseminated actinic porokeratosis', 'drug eruption', 'dry skin eczema', 'dyshidrosiform eczema', 'dysplastic nevus', 'eccrine poroma', 'eczema', 'epidermal nevus', 'epidermoid cyst', 'epithelioma adenoides cysticum', 'erythema ab igne', 'erythema annulare centrifigum', 'erythema craquele', 'erythema multiforme', 'exfoliative erythroderma', 'factitial dermatitis', 'favre racouchot', 'fibroma', 'fibroma molle', 'fixed drug eruption', 'follicular mucinosis', 'follicular retention cyst', 'fordyce spots', 'frictional lichenoid dermatitis', 'ganglion', 'geographic tongue', 'granulation tissue', 'granuloma annulare', 'green nail', 'guttate psoriasis', 'hailey hailey disease', 'half and half nail', 'halo nevus', 'herpes simplex virus', 'herpes zoster', 'hidradenitis suppurativa', 'histiocytosis x', 'hyperkeratosis palmaris et plantaris', 'hypertrichosis', 'ichthyosis', 'impetigo', 'infantile atopic dermatitis', 'inverse psoriasis', 'junction nevus', 'keloid', 'keratoacanthoma', 'keratolysis exfoliativa of wende', 'keratosis pilaris', 'kerion', 'koilonychia', 'kyrle s disease', 'leiomyoma', 'lentigo maligna melanoma', 'leukocytoclastic vasculitis', 'leukonychia', 'lichen planus', 'lichen sclerosis et atrophicus', 'lichen simplex chronicus', 'lichen spinulosis', 'linear epidermal nevus', 'lipoma', 'livedo reticularis', 'lymphangioma circumscriptum', 'lymphocytic infiltrate of jessner', 'lymphomatoid papulosis', 'mal perforans', 'malignant melanoma', 'median nail dystrophy', 'melasma', 'metastatic carcinoma', 'milia', 'molluscum contagiosum', 'morphea', 'mucha habermann disease', 'mucous membrane psoriasis', 'myxoid cyst', 'nail dystrophy', 'nail nevus', 'nail psoriasis', 'nail ridging', 'neurodermatitis', 'neurofibroma', 'neurotic excoriations', 'nevus comedonicus', 'nevus incipiens', 'nevus sebaceous of jadassohn', 'nevus spilus', 'nummular eczema', 'onychogryphosis', 'onycholysis', 'onychomycosis', 'onychoschizia', 'paronychia', 'pearl penile papules', 'perioral dermatitis', 'pincer nail syndrome', 'pitted keratolysis', 'pityriasis alba', 'pityriasis rosea', 'pityrosporum folliculitis', 'poikiloderma atrophicans vasculare', 'pomade acne', 'pseudofolliculitis barbae', 'pseudorhinophyma', 'psoriasis', 'pustular psoriasis', 'pyoderma gangrenosum', 'pyogenic granuloma', 'racquet nail', 'radiodermatitis', 'rhinophyma', 'rosacea', 'scalp psoriasis', 'scar', 'scarring alopecia', 'schamberg s disease', 'sebaceous gland hyperplasia', 'seborrheic dermatitis', 'seborrheic keratosis', 'skin tag', 'solar lentigo', 'stasis dermatitis', 'stasis edema', 'stasis ulcer', 'steroid acne', 'steroid striae', 'steroid use abusemisuse dermatitis', 'stomatitis', 'strawberry hemangioma', 'striae', 'subungual hematoma', 'syringoma', 'terry s nails', 'tinea corporis', 'tinea cruris', 'tinea faciale', 'tinea manus', 'tinea pedis', 'tinea versicolor', 'toe deformity', 'trichilemmal cyst', 'trichofolliculoma', 'trichostasis spinulosa', 'ulcer', 'urticaria', 'varicella', 'verruca vulgaris', 'vitiligo', 'wound infection', 'xerosis']

const MODEL_PATH = './assets/web_model/tensorflowjs_model.pb'
const WEIGHTS_PATH = './assets/web_model/weights_manifest.json'
const IMAGE_SIZE = 224;
const TOPK_PREDICTIONS = 5;
const INPUT_NODE_NAME = 'input';
const OUTPUT_NODE_NAME = 'final_result';
const PREPROCESS_DIVISOR = tfc.scalar(255 / 2);

export class MobileNet {
  constructor() {}

  async load() {
    this.model = await loadFrozenModel(MODEL_PATH, WEIGHTS_PATH);
  }

  dispose() {
    if (this.model) {
      this.model.dispose();
    }
  }
  /**
   * Infer through MobileNet. This does standard ImageNet pre-processing before
   * inferring through the model. This method returns named activations as well
   * as softmax logits.
   *
   * @param input un-preprocessed input Array.
   * @return The softmax logits.
   */
  predict(input) {
    console.log(input.shape)
    const preprocessedInput = tfc.div(
        tfc.sub(input.asType('float32'), PREPROCESS_DIVISOR),
        PREPROCESS_DIVISOR);
    console.log(preprocessedInput.shape)
    const reshapedInput =
        preprocessedInput.reshape([1, ...preprocessedInput.shape]);
    const dict = {};
    dict[INPUT_NODE_NAME] = reshapedInput;
    console.log(reshapedInput.shape)
    return this.model.execute(dict, OUTPUT_NODE_NAME);
  }

  getTopKClasses(predictions, topK) {
    const values = predictions.dataSync();
    predictions.dispose();

    let predictionList = [];
    for (let i = 0; i < values.length; i++) {
      predictionList.push({value: values[i], index: i});
    }
    predictionList = predictionList
                         .sort((a, b) => {
                           return b.value - a.value;
                         })
                         .slice(0, topK);

    return predictionList.map(x => {
      return {label: classes[x.index], value: x.value};
    });
  }
}

const webcam = new Webcam(document.getElementById('webcam'));
const mobileNet = new MobileNet();

(async function main() {
  try {
    alert("IMPORTANT: I do not store any images or video. Honestly I can't afford to do that. All inference is happening on-device and nothing is sent to me.");
    await webcam.setup();
    console.time('Loading of model');
    await mobileNet.load();
    console.timeEnd('Loading of model');
    doneLoading();

    mobileNet.predict(tf.zeros([IMAGE_SIZE, IMAGE_SIZE, 3])).dispose();

  } catch(e) {
    console.error(e);
  }
})();

function doneLoading() {
  const elem = document.getElementById('loading-message');
  elem.style.display = 'none';

  const successElem = document.getElementById('capture-button');
  successElem.style.display = 'inline-block';

  const webcamElem = document.getElementById('webcam-wrapper');
  webcamElem.style.display = 'flex';
}

document.getElementById('capture-button').addEventListener('click', async () => {
  console.log('pic taken');
  const img = webcam.capture();
  let result = mobileNet.predict(img);
  const topK = mobileNet.getTopKClasses(result, 5);
  console.log(topK);
});