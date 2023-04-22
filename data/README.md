### Dataset

#### Generating the data

For the `historical_data.csv`, simply run

```
    python data/make_data.py --filename data/historical_data.csv
```

For the `latest_data.csv`, simply run

```
    python data/make_data.py --filename data/latest_data.csv --n-samples 200 --noise 0.3
```

#### Getting the data

To get the data directly, simply run

```
dvc pull
```