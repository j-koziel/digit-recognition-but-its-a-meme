import { MnistData } from "./data.js";
import showAccuracy from "./showAccuracy.js";
import showConfusion from "./showConfusion.js";
import showExamples from "./showExamples.js";
import getModel from "./getModel.js";
import train from "./train.js";

async function run() {
  const data = new MnistData();
  await data.load();
  await showExamples(data);
  const model = getModel();
  tfvis.show.modelSummary({ name: "Model Architecture", tab: "Model" }, model);
  await train(model, data);
  await showAccuracy(model, data);
  await showConfusion(model, data);
}

run();

document.addEventListener("click", run);
