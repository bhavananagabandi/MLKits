require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs');
const loadSCV = require('./load-csv');
const LinearRegression = require('./linear-regression');

let { features, labels, testFeatures, testLabels } = loadSCV('./cars.csv', {
    shuffle: true,
    splitTest: 50,
    dataColumns: ['horsepower', 'weight', 'displacement'],
    labelColumns: ['mpg'] 
});

// features = tf.tensor(features)
const regression = new LinearRegression(features, labels, {
    learningRate: 0.1,
    iterations: 100
})

regression.train();
const r2 = regression.test(testFeatures, testLabels);
console.log('R2 is', r2)