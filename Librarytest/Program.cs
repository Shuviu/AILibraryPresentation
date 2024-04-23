using AILibrary;

SpiralDatasetGen spiralDatasetGen = new SpiralDatasetGen(1, 6.5);
spiralDatasetGen.GenerateSamples();

Dataset dataset = new Dataset(spiralDatasetGen.GetSamples(), spiralDatasetGen.GetLabels());

IOptimizer optimizer = new SGDOptim(0.01);
ILossFunction lossFunction = new CategoricalCrossEntropyLoss();

Trainer trainer = new Trainer(lossFunction, optimizer);
trainer.UpdateDataset(dataset);

trainer.AddNeuronLayer(2, 4);
trainer.AddNeuronLayer(4, 4);
trainer.AddNeuronLayer(4, 2);

trainer.Train(4000);