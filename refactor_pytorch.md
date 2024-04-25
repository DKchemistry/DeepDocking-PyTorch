# Refactoring DD from TF to PyTorch 

## What phases and files interact with Tensorflow that I will need to change? 

### Phase 4

`phase_4.sh` will call `scripts_2/simple_job_models.py`, this in turn will write a series of jobs with hyperparameters (using grid search) to run `progressive_docking.py`. This is what handles the actual tensorflow model that is used. I can see lines like: 

```py
prediction_valid = progressive_docking.predict(X_valid)
```

Indicating `progessive_docking` is a class of some sort, we can see it instantiated here: 

```py
from ML.DDMetrics import *
    metrics = ['accuracy', tf.keras.metrics.Recall(), tf.keras.metrics.Precision()]
    progressive_docking = DDModel(mode='original',
                                  input_shape=Oversampled_X_train.shape[1:],
                                  hyperparameters=hyperparameters,
                                  metrics=metrics)
progressive_docking.model.summary()
```

Here we are importing all of the ML.DDMetrics functions (I would like to generally refactor this to avoid name space pollution). 

Then we have progressive_docking as an instance of `DDModel` with some arguments: 

1. The input shape of our NN, `Oversampled_X_train.shape[1:]`

* Whatever the exact datastructure of that is, we are skipping element 0 and continuining onwards. 

2. the hyperparameters (likely set from the for loops we see in in `simple_jobs.py`).

* This will likely need some re-work as well. 

3. the metrics, sourced from `ML.DDMetrics`. The wildcard import should cover all functions. 

After getting a quick scope of what is going on, it is not going to be trivial to adapt TF -> PyTorch is some sort of piece by piece fashion. 

Instead, I think we need some sort of iterative design in continous consultation with Francesco. 

The first thing is to figure out the model used. 

```py
def original(self, input_shape):
        x_input = Input(input_shape, name="original")
        x = x_input
        for j, i in enumerate(self.hyperparameters['bin_array']):
            if i == 0:
                x = Dense(self.hyperparameters['num_units'], name="Hidden_Layer_%i" % (j + 1))(x)
                x = BatchNormalization()(x)
                x = Activation('relu')(x)
            else:
                x = Dropout(self.hyperparameters['dropout_rate'])(x)
        x = Dense(1, activation=self.output_activation, name="Output_Layer")(x)
        model = Model(inputs=x_input, outputs=x, name='Progressive_Docking')
        return model
```

We should be able to keep the `simple_job/` dir and it's creation intact, we simply need to be able to call the different arguments involved. 

If we look at what metrics are ultimately reported, we see: 

```sh
---------------
Hyperparameters:
- Model Number: 2 # model id so we can find it
- Training Time: 565.355 # timing function
  - OS: 10 # oversampling, not sure how this is done
  - Batch Size: 256 # batch size, handled via torch.DataSets and torch.DataLoader
  - Learning Rate: 0.0001 # learning rate, standard hyperparam
  - Bin Array: 2 # number of hidden layers
  - Num. Units: 1500 # number of nuerons per layer
  - Dropout Freq.: 0.2 # i know this has to do with dropped neurons...

  - Class Weight Parameter wt: 3.0 # not sure what this is
  - cf: -9.236514914296889 # cut off for label classification
  - auc vl: 0.9107155386224814 # roc auc for valid?
  - auc te: 0.9134509600982472 # roc auc test?
  - Precision validation: 0.03501034473950839 # straightforward
  - Precision testing: 0.035064283100904474 # straightforward
  - Recall testing: 0.9097130242825607 # need to find how he sets the recall
  - Pos ct orig: 4530 # sum of y_valid, probably gets smaller per iter
                      # yes it does 
  - Total Left: 16866695.46766841 # this has something to do with predicting on te/vl
  - Total Left testing: 16888082.630775496 # this as well, must be testing

---------------
```

While it is not good practice, I could really condense DDModel/DDMetrics as one file? Though seperating them out isn't very hard. 

In order to test any of this, we are going to have to get some data to work with. I previously ran the original benchmarking on 2RH1, where is that?

/mnt/data/dk/work/DeepDocking/projects/2RH1

(remember to have Quartz running w/ X11 forwarding for pbcopy to work, think it needs the terminal to be active)

let's make

/mnt/data/dk/work/DeepDocking/projects/pytorch_2RH1

let's edit that log file immediately 

```sh
/mnt/data/dk/work/DeepDocking/projects
pytorch_2RH1 # done 
/mnt/data/dk/work/grids/2RH1_grid_MinimizedWithLigand/2RH1_grid_MinimizedWithLigand.zip
/mnt/data/dk/work/DeepDocking/library_prepared_fp
/mnt/data/dk/work/DeepDocking/library_prepared
Glide
24
462000
/mnt/data/dk/work/DD_protocol/scripts_1/2RH1_glide_template.in
```

Now we can try running `progressive_docking_pytorch.py` and see what happens. 

Let's grab the args from a generic simple jobs run: 


python -u progressive_docking_pytorch.py -os 10 -bs 256 -num_units 100 -dropout 0.2 -learn_rate 0.0001 -bin_array 2 -wt 3 -cf -9.236514914296889 -rec 0.9 -n_it 1 -t_mol 65.994673 --data_path /mnt/data/dk/work/DeepDocking/projects/pytorch_2RH1 --save_path /mnt/data/dk/work/DeepDocking/projects/pytorch_2RH1 -n_mol 462000

I am cutting down the code from `progressive_doing.py` to just the bare essentials. 

* smiles is not used 
* continous is not used 
* normalize is not used 
* i dont think encode smiles is used 

getting the data in is really hard. i have to sit down and really understand the code to do that. 

### Phase 5