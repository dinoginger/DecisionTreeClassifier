"""
Decision Trees Task Code by Roman Mutel & Marko Ruzak
"""

from iris import iris
from numpy import array
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

class Node:
    def __init__(self, X, y, gini):
        self.X = X  # всі 150 квіток -> 150 лістів з чотирма параметрами
        self.y = y  # відповідно до Х, якого виду кожна квітка (1 або 2 або 3)
        self.gini = gini  # потрібне для перевірки на листок (передається від батька)
        self.feature_index = 0  # за якою ознакою порівнювали X[n][0 або 1, 2, 3...]
        self.threshold = 0  # число для порівняння
        self.left = None
        self.right = None
        self.flower = None


class MyDecisionTreeClassifier:
    def __init__(self, max_depth=20):
        self.root = None
        self.max_depth = max_depth  # задавати максимальну висоту дерева.
    
    def gini(self, groups, classes):
        """
        A Gini score gives an idea of how good a split is by how mixed the
        classes are in the two groups created by the split.
        
        A perfect separation results in a Gini score of 0,
        whereas the worst case split that results in 50/50/50
        classes in each group result in a Gini score of 0.67

        Args:
            groups (list[list[array]]): left and right subgroups after split procedure
            classes (list): classes ([0,1,2] for our iris dataset)

        Returns:
            float: gini value for given group (from 0 to 1)
        """

        entries = sum([len(group) for group in groups]) # 150 for our iris dataset on first iter
        gini_value = 0

        for group in groups:
            local_gini = 1
            group_entries = len(group)
            if group_entries == 0:
                continue

            for flower in classes:
                # each row contains 4 float numbers as first item and flower class as second item 
                # (so we have to take entry[-1])
                local_gini -= ([entry[-1] for entry in group].count(flower) / group_entries) ** 2
            gini_value += local_gini * group_entries / entries

        return gini_value
    
    def split_data(self, X, y) -> 'tuple[int, int]':
        """
        Сплітаємо дані в два лісти, за якимось значенням по одному з чотирьох параметрів.
        Іншими словами, ми ітеруємо наш ліст з елементами, в яких параметрами кожної квітки,
        (в кожному елементі 4 параметри), та сплітаємо, по суті, за кожним параметром
        кожної квітки.

        Після цього обчислюємо Gini індекс, та обираємо найкращий варіант спліту.

        :param X: Датасет з параметрами всіх квіток List[list[float, float, float, float]]
        :param y: Список у якому типи всіх квіток (1, 2 або 3) у відповідності до першого датасету.
        :return: Повертає індекс і threshold value - індекс це один з чотирьох параметрів,
        за яким порівнююьтся квітки, а threshold value - значення параметру одної з квіток
        (тої, за якою і порівнювався (і сплітився кінцево) датасет)
        """
            # test all the possible splits in O(N^2)
            # return index and threshold value
        best_gini = float("inf")
        for index in range(len(X[0])):
            for for_value in X:
                left, right = list(), list()
                for i,row in enumerate(X):
                    if row[index] <= for_value[index]:
                        left.append((row, y[i]))
                    else:
                        right.append((row, y[i]))
                groups = [left, right]

                gini_value = self.gini(groups, list(set(y)))
                # print(gini_value)
                if gini_value == 0:
                    return index, for_value[index], gini_value, groups
                if gini_value < best_gini:
                    best_gini = gini_value
                    best_index = index
                    best_value = for_value[index]
                    best_groups = groups

        return best_index, best_value, best_gini, best_groups

    def build_tree(self, X, y, depth = 0, terminal = False):
        # create a root node
        # recursively split until max depth is not exeeced

        # continue splitting
        best_index, best_value, best_gini, (left, right) = self.split_data(X, y)
        left_y = [entry[1] for entry in left]
        right_y = [entry[1] for entry in right]
        left = [entry[0] for entry in left]
        right = [entry[0] for entry in right]

        # 'and len(set(y)) == 1' fixed the problem with gini = 0 when there are still two types of flower
        # if gini == 0, it either means that we already separated all the flowers or we can separate them
        # on the next split
        if (best_gini == 0 and len(set(y)) == 1) or depth == self.max_depth:
            node = Node(X, y, best_gini)
            node.flower = max([(flower, y.count(flower)) for flower in set(y)], key=lambda x:x[1])[0]
            # print(f'Added leaf flower (class): {node.flower}, depth: {depth}, gini: {node.gini}, y: {[(flower, y.count(flower)) for flower in set(y)]}')
            return node

        node = Node(X, y, best_gini)
        node.feature_index = best_index
        node.threshold = best_value
        node.flower = max([(flower, y.count(flower)) for flower in set(y)], key=lambda x:x[1])[0]
        node.left = self.build_tree(left, left_y, depth + 1)
        node.right = self.build_tree(right, right_y, depth + 1)

        # print(f'Added node feature index:{node.feature_index}, threshold: {node.threshold}, flower (class): {node.flower}, gini: {node.gini}, depth: {depth}')
        # print(f'y: {[(flower, y.count(flower)) for flower in set(y)]}')
        return node

    def fit(self, X, y):
        # basically wrapper for build tree
        X = X.tolist()
        y = y.tolist()
        self.root = self.build_tree(X, y)
        return

    def predict_case(self, X_test):
        node = self.root
        while node.left:
            if X_test[node.feature_index] < node.threshold:
                node = node.left
            else:
                node = node.right
        return node.flower

    def predict(self, X_test):

        # traverse the tree while there is left node
        # and return the predicted class for it, 
        # note that X_test can be not only one example

        if isinstance(X_test[0], float):
            return self.predict_case(X_test)
        y_test = list()
        for element in X_test:
            y_test.append(self.predict_case(element))
        return y_test


if __name__ == '__main__':
    iris = load_iris()
    m_t = MyDecisionTreeClassifier(10)
    X, y = iris.data, iris.target
    X, X_test, y, y_test = train_test_split(X, y, test_size= 0.20)
    root = m_t.fit(X, y)
    print('Generated flowers')
    print(y_test.tolist())
    print('Predicted flowers')
    predictions = m_t.predict(X_test)
    print(predictions)
    print('Model Accuracy is: ')
    print(sum(predictions == y_test) / len(y_test))
