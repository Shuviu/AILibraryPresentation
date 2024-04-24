using AILibrary;

SpiralDatasetGen spiralDatasetGen = new SpiralDatasetGen(1, 0.5);
spiralDatasetGen.GenerateSamples(2);

Dataset dataset = new Dataset(spiralDatasetGen.GetSamples(), spiralDatasetGen.GetLabels());

IOptimizer optimizer = new SGDOptim(0.1);
ILossFunction lossFunction = new CategoricalCrossEntropyLoss();

Trainer trainer = new Trainer(lossFunction, optimizer);
trainer.UpdateDataset(dataset);

trainer.AddNeuronLayer(new NeuronLayer(2, 16));
trainer.AddNeuronLayer(new SigmoidActivation());
trainer.AddNeuronLayer(new NeuronLayer(16, 16));
trainer.AddNeuronLayer(new SigmoidActivation());
trainer.AddNeuronLayer(new NeuronLayer(16, 16));
trainer.AddNeuronLayer(new ReLUActivation());
trainer.AddNeuronLayer(new NeuronLayer(16, 16));
trainer.AddNeuronLayer(new SigmoidActivation());
trainer.AddNeuronLayer(new NeuronLayer(16, 2));

trainer.Train(40);