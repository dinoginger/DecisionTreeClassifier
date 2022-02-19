"""
Decision Trees Task Code by Roman Mutel & Marko Ruzak
"""



class Node:
    def __init__(self, X, y, gini):
        self.X = X  # всі 150 квіток -> 150 лістів з чотирма параметрами
        self.y = y  # відповідно до Х, якого виду кожна квітка (1 або 2 або 3)
        self.gini = gini  # потрібне для перевірки на листок (передається від батька)
        self.feature_index = 0  # за якою ознакою порівнювали X[n][0 або 1, 2, 3...]
        self.threshold = 0  # число для порівняння
        self.left = None
        self.right = None


class MyDecisionTreeClassifier:
    
    def __init__(self, max_depth):
        self.max_depth = max_depth  # задавати максимальну висоту дерева.
    
    def gini(self, groups, classes):
        """
        A Gini score gives an idea of how good a split is by how mixed the
        classes are in the two groups created by the split.
        
        A perfect separation results in a Gini score of 0,
        whereas the worst case split that results in 50/50
        classes in each group result in a Gini score of 0.5
        (for a 2 class problem).
        """

        pass
    
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
                for row in X:
                    if row[index] <= for_value[index]:
                        left.append(row)
                    else:
                        right.append(row)
                groups = [left, right]

                gini = self.gini(groups, y)
                if gini == 0:
                    return index, for_value[index]
                if gini < best_gini:
                    best_gini = gini

    def build_tree(self, X, y, depth = 0):
        
        # create a root node
        
        
        # recursively split until max depth is not exeeced
        
        pass
    
    def fit(self, X, y):
        
        # basically wrapper for build tree

        # Sort X here before passing it to data split method!
        pass
    
    def predict(self, X_test):
        
        # traverse the tree while there is left node
        # and return the predicted class for it, 
        # note that X_test can be not only one example
        
        pass
