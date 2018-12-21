import numpy as np
import pandas as pd

print(pd.__version__)

cityName = pd.Series(["San Francisco", "San Jose", "Sacramento"])
cityPoplation = pd.Series([852469, 1015785, 485199])

dataFrame = pd.DataFrame({"City Name": cityName, "Population": cityPoplation})

# print(dataFrame);


california_housing_dataframe = pd.read_csv(
    "../california_housing_train.csv", sep=",")

california_housing_dataframe.describe()
# print(california_housing_dataframe.describe());

california_housing_dataframe.head()
# print(california_housing_dataframe.head(100));

california_housing_dataframe.hist('housing_median_age')

cities = pd.DataFrame({'City name': cityName, 'Population': cityPoplation})

print(type(cities["City name"]))
cities["City name"]
# p rint(cities["City name"]);


print(type(cities["City name"][1]))
cities["City name"][1]
# print(cities["City name"][1]);

print(type(cities["City name"][0:2]))
cities["City name"][0:2]
# print(cities["City name"][0:2]);

# print(cityPoplation / 10);


a = np.log(cityPoplation)
# print(a);

cityPoplation.apply(lambda val: val > 100000)
# print(cityPoplation.apply(lambda val : val > 100000));

cities["Area square miles"] = pd.Series([46.87, 176.53, 97.92])
cities["Population density"] = cities["Population"] / \
    cities["Area square miles"]
cities
# print(cities);

cities["Is wide and has saint name"] = cities["Area square miles"].apply(
    lambda area: area > 50) & cities["City name"].apply(lambda name: name.startswith("San"))

# print(cities);

print(cityName.index)
print(cities.index)

print(cities)
# print(cities.reindex([2, 0, 1]));

print(cities.reindex(np.random.permutation(cities.index)))

d = cities.reindex([5, 6, 4, 2, 3, 1, 0])
print(cities)
print(d)

print(type(d))
