require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs');
const loadSCV = require('./load-csv');
const LinearRegression = require('./linear-regression');

let { features, labels, testFeatures, testLabels } = loadSCV('./cars.csv', {
    shuffle: true,
    splitTest: 50,
    dataColumns: ['horsepower'],
    labelColumns: ['mpg'] 
});

// features = tf.tensor(features)
const regression = new LinearRegression(features, labels, {
    learningRate: 0.0001,
    iterations: 100
})

regression.train();

console.log('UPdated M is: ', regression.m, 'Updated B is: ', regression.b);
