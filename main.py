"""
Decision Trees Task Code by Roman Mutel & Marko Ruzak
"""

from sklearn.datasets import load_iris
from iris import iris

# iris = load_iris()

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

    def build_tree(self, X, y, depth = 0):
        # create a root node
        # recursively split until max depth is not exeeced

        # continue splitting
        best_index, best_value, best_gini, (left, right) = self.split_data(X, y)
        left = [entry[0] for entry in left]
        right = [entry[0] for entry in right]

        if best_gini == 0  or depth == self.max_depth:
            node = Node(X, y, 0)
            node.flower = max([(flower, y.tolist().count(flower)) for flower in set(y)], key=lambda x:x[1])[0]
            return

        node = Node(X, y, best_gini)
        node.feature_index = best_index
        node.threshold = best_value
        node.flower = max([(flower, y.tolist().count(flower)) for flower in set(y)], key=lambda x:x[1])[0]
        node.left = self.build_tree(left, y[:len(left)], depth + 1)
        node.right = self.build_tree(right, y[len(left):], depth + 1)

        return node

    
    def fit(self, X, y):
        # basically wrapper for build tree
        root = self.build_tree(X, y)
        return root
    
    def predict(self, X_test):
        
        # traverse the tree while there is left node
        # and return the predicted class for it, 
        # note that X_test can be not only one example
        
        pass

if __name__ == '__main__':
    m_t = MyDecisionTreeClassifier(5)
    root = m_t.fit(iris[0], iris[1])
    print(root.right.gini)
    # print(m_t.split_data(iris[0], iris[1]))
