from utils import *


class KnnImpute():
    def __init__(self, k=3, threshold=0.4):
        self.k = k
        self.threshold = threshold

    def fit_transform(self, target_attr, fillin_attr, df):
        self.target_attr, self.fillin_attr, df = target_attr, fillin_attr, df
        self.higher_thre_ins, self.target_nan, mostly_nan = self._filtering(df)
        self._random_assign_num(mostly_nan)
        nomial_col, _ = self._detect_df_columns_type()
        self.nomial_col_index = []
        for i in self.fillin_attr:
            if(i in nomial_col):
                self.nomial_col_index.append(self.fillin_attr.index(i))
        X = df[self.fillin_attr].values
        # calculate distance
        distance_matrix = np.zeros((len(self.target_nan), X.shape[0]))
        for i in range(len(self.target_nan)):
            a = X[self.target_nan[i]][:]
            for j in range(X.shape[0]):
                if(j in self.higher_thre_ins or j in self.target_nan):
                    distance_matrix[i][j] = 10000
                    continue
                distance = 0
                distance_cat = 0
                distance_num = 0
                b = X[j][:]
                count = 0
                for z in range(X.shape[1]):
                    if(math.isnan(a[z]) and math.isnan(b[z])):
                        distance += 0
                    elif(math.isnan(a[z]) or math.isnan(b[z])):
                        count += 1
                    elif(z in self.nomial_col_index):
                        # Euclidean distance
                        distance_cat += abs(float(a[z])-float(b[z]))
                    else:
                        # Mannhantan distance
                        distance_num += abs(float(a[z])-float(b[z]))**2
                distance = distance_cat + np.sqrt(distance_num)
                distance += count*math.ceil(distance/X.shape[1])
                distance_matrix[i][j] = distance
        # rank the distance
        X_ = df[self.target_attr].values
        for i in range(distance_matrix.shape[0]):
            sort_distance = np.argsort(distance_matrix[i])
            k_neighbor = sort_distance[1:self.k+1]
            k_neighbor_values = []
            for j in k_neighbor:
                k_neighbor_values.append(X_[j])
            impute_value = np.median(k_neighbor_values)
            df.loc[self.target_nan[i], self.target_attr] = impute_value
        self.df = df
        return df[self.target_attr]

    def transfrom(self, df):
        _, target_nan, mostly_nan = self._filtering(df)
        X, X_pool = df[self.fillin_attr].values, self.df[self.fillin_attr].values
        # calculate distance
        distance_matrix = np.zeros((len(target_nan), X_pool.shape[0]))
        for i in range(len(target_nan)):
            a = X[target_nan[i]][:]
            for j in range(X_pool.shape[0]):
                if(j in self.higher_thre_ins or j in self.target_nan):
                    distance_matrix[i][j] = 10000
                    continue
                distance = 0
                distance_cat = 0
                distance_num = 0
                b = X_pool[j][:]
                count = 0
                for z in range(X_pool.shape[1]):
                    if(math.isnan(a[z]) and math.isnan(b[z])):
                        distance += 0
                    elif(math.isnan(a[z]) or math.isnan(b[z])):
                        count += 1
                    elif(z in self.nomial_col_index):
                        # Euclidean distance
                        distance_cat += abs(float(a[z])-float(b[z]))
                    else:
                        # Mannhantan distance
                        distance_num += abs(float(a[z])-float(b[z]))**2
                distance = distance_cat + np.sqrt(distance_num)
                distance += count*math.ceil(distance/X.shape[1])
                distance_matrix[i][j] = distance
        # rank the distance
        X_ = df[self.target_attr].values
        for i in range(distance_matrix.shape[0]):
            sort_distance = np.argsort(distance_matrix[i])
            k_neighbor = sort_distance[1:self.k+1]
            k_neighbor_values = []
            for j in k_neighbor:
                k_neighbor_values.append(X_[j])
            impute_value = np.median(k_neighbor_values)
            df.loc[target_nan[i], self.target_attr] = impute_value
        return df[self.target_attr]

    def _filtering(self, df):
        # filter instances by threshold and check the instances that should be imputed
        higher_thre_ins = []
        target_nan = []
        mostly_nan = []
        for i in df.index:
            count = 0
            for j in self.fillin_attr:
                if(math.isnan(df.loc[i, j])):
                    count += 1
            count /= len(self.fillin_attr)
            if(count > 0.5):
                mostly_nan.append(i)
            if(math.isnan(df.loc[i, self.target_attr])):
                target_nan.append(i)
            if(count >= self.threshold):
                higher_thre_ins.append(i)
        return higher_thre_ins, target_nan, mostly_nan

    def _random_assign_num(self, mostly_nan):
        # random assign fillin_attr mostly nan attributes
        for i in mostly_nan:
            fill_c = np.random.choice(
                self.fillin_attr, math.floor(len(self.fillin_attr)*0.5))
            for j in fill_c:
                values_list = [z for z in self.df[j].unique() if(
                    math.isnan(z) == False)]
                self.df.loc[i, j] = np.random.choice(values_list, 1)[0]

    def _detect_df_columns_type(self):
        nomial_col = []
        num_col = []
        for i in self.fillin_attr:
            if(len(self.df[i].value_counts()) < self.threshold):
                nomial_col.append(i)
            else:
                num_col.append(i)
        return nomial_col, num_col
