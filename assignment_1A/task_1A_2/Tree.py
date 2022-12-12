""" This file created as supplementary code for tree-related questions in DD2434 - Assignment 2.
    Current version of the document is built up on 2018, 2019 and 2020 codes,
    accessible via: https://gits-15.sys.kth.se/butepage/MLadvHT18, https://gits-15.sys.kth.se/koptagel/AdvML19 and https://gits-15.sys.kth.se/koptagel/AdvML20 """

import numpy as np
import pickle


class Node:
    """ Node Class
        Class for tree nodes. Each node has a name, a list of categorical distribution probabilities (thetas),
        an ancestor node and the list of children nodes. """

    def __init__(self, name, cat):
        self.name = name
        self.cat = []
        for c in cat:
            self.cat.append(c)
        self.ancestor = None
        self.descendants = []

    def print(self):
        """ This function prints the node's information. """

        if self.ancestor is None:
            print("\tNode: ", self.name, "\tParent: ", self.ancestor, "\tNum Children: ", len(self.descendants),
                  "\tCat: ", self.cat)
        else:
            print("\tNode: ", self.name, "\tParent: ", self.ancestor.name, "\tNum Children: ", len(self.descendants),
                  "\tCat: ", self.cat)


class Tree:
    """ Tree Class
        Class for tree structures. Each tree has a root node, the number of nodes, the number of leaves,
        k (the number of possible values), alpha for dirichlet prior to categorical distributions,
        the number of samples, the list of samples
        and the list of filtered samples (inner node values are replaced with np.nan). """

    def __init__(self):
        self.root = None
        self.num_nodes = 0
        self.num_leaves = 0
        self.k = 0
        self.alpha = []
        self.num_samples = 0
        self.samples = []
        self.filtered_samples = []
        self.newick = ""

    def create_random_tree(self, seed_val, k, max_num_nodes=10, max_branch=5, alpha=None):
        """ This function creates a random tree. """

        if alpha is None:
            alpha = []

        print("Creating random tree...")
        np.random.seed(seed_val)

        if len(alpha) == 0:
            alpha = [1.0] * k
        elif len(alpha) != k or np.sum(np.array(alpha) < 0) != 0:
            print("Error! Alpha needs to contain k positive values! ")
            return None

        self.root = Node(str(0), np.random.dirichlet(alpha))
        visit_list = [self.root]

        num_nodes = 1
        num_leaves = 1
        while len(visit_list) != 0 and num_nodes < max_num_nodes:
            cur_node = visit_list[0]
            visit_list = visit_list[1:]

            if cur_node == self.root:
                num_children = np.random.randint(1, max_branch + 1)
            else:
                num_children = np.random.randint(0, max_branch + 1)

            if num_children > 0:
                num_leaves = num_leaves + num_children - 1
                children_list = []
                for i in range(num_children):
                    cat = []
                    for theta in range(k):
                        cat.append(np.random.dirichlet(alpha))
                    child_node = Node(str(num_nodes), cat)
                    child_node.ancestor = cur_node
                    children_list.append(child_node)
                    visit_list.append(child_node)
                    num_nodes = num_nodes + 1
                cur_node.descendants = children_list

        self.num_leaves = num_leaves
        self.num_nodes = num_nodes
        self.k = k
        self.alpha = alpha
        self.newick = self.get_tree_newick()

    def create_random_tree_fix_nodes(self, seed_val, k, num_nodes=10, max_branch=5, alpha=None):
        """ This function creates a random tree. """

        if alpha is None:
            alpha = []

        print("Creating random tree with fixed number of nodes...")
        np.random.seed(seed_val)

        if len(alpha) == 0:
            alpha = [1.0] * k
        elif len(alpha) != k or np.sum(np.array(alpha) < 0) != 0:
            print("Error! Alpha needs to contain k positive values! ")
            return None

        self.root = Node(str(0), np.random.dirichlet(alpha))
        visit_list = [self.root]

        cur_num_nodes = 1
        num_leaves = 1
        while cur_num_nodes != num_nodes:  # len(visit_list) != 0 and cur_num_nodes < num_nodes:
            cur_node = np.random.choice(visit_list)

            if cur_node == self.root:
                num_children = np.random.randint(1, min(max_branch + 1, num_nodes - cur_num_nodes + 1))
            else:
                num_children = np.random.randint(0, min(max_branch + 1, num_nodes - cur_num_nodes + 1))

            if num_children > 0:
                visit_list.remove(cur_node)
                num_leaves = num_leaves + num_children - 1
                children_list = []
                for i in range(num_children):
                    cat = []
                    for theta in range(k):
                        cat.append(np.random.dirichlet(alpha))
                    child_node = Node(str(cur_num_nodes), cat)
                    child_node.ancestor = cur_node
                    children_list.append(child_node)
                    visit_list.append(child_node)
                    cur_num_nodes = cur_num_nodes + 1
                cur_node.descendants = children_list

        self.num_leaves = num_leaves
        self.num_nodes = cur_num_nodes
        self.k = k
        self.alpha = alpha
        self.newick = self.get_tree_newick()

    def create_random_binary_tree(self, seed_val, k, num_nodes=10, alpha=None):
        """ This function creates a random binary tree. """

        if alpha is None:
            alpha = []

        print("Creating random binary tree with fixed number of nodes...")
        np.random.seed(seed_val)

        if num_nodes % 2 != 1:
            print("\tWarning! Specified num_nodes (%d) is not enough to generate a binary tree. "
                  "num_nodes is changed to: %d" % (num_nodes, num_nodes + 1))
            num_nodes = num_nodes + 1

        if len(alpha) == 0:
            alpha = [1.0] * k
        elif len(alpha) != k or np.sum(np.array(alpha) < 0) != 0:
            print("Error! Alpha needs to contain k positive values! ")
            return None

        self.root = Node(str(0), np.random.dirichlet(alpha))
        visit_list = [self.root]

        cur_num_nodes = 1
        num_leaves = 1
        while cur_num_nodes < num_nodes:
            cur_node = np.random.choice(visit_list)
            if cur_node == self.root:
                num_children = 2
            else:
                num_children = np.random.choice([0, 2], p=[0.5, 0.5])

            if num_children > 0:
                num_leaves = num_leaves + num_children - 1
                visit_list.remove(cur_node)
                children_list = []
                for i in range(num_children):
                    cat = []
                    for theta in range(k):
                        cat.append(np.random.dirichlet(alpha))
                    child_node = Node(str(cur_num_nodes), cat)
                    child_node.ancestor = cur_node
                    children_list.append(child_node)
                    visit_list.append(child_node)
                    cur_num_nodes = cur_num_nodes + 1
                cur_node.descendants = children_list

        self.num_leaves = num_leaves
        self.num_nodes = cur_num_nodes
        self.k = k
        self.alpha = alpha
        self.newick = self.get_tree_newick()

    def sample_tree(self, num_samples=1, seed_val=None):
        """ This function generates samples from the tree. """

        print("Sampling tree nodes...")
        if seed_val is not None:
            np.random.seed(seed_val)

        samples = np.zeros((num_samples, self.num_nodes))
        samples[:] = np.nan
        filtered_samples = np.zeros((num_samples, self.num_nodes))
        filtered_samples[:] = np.nan

        if self.num_nodes > 0:

            for sample_idx in range(num_samples):
                visit_list = [self.root]

                while len(visit_list) != 0:
                    cur_node = visit_list[0]
                    visit_list = visit_list[1:] + cur_node.descendants
                    par_node = cur_node.ancestor

                    if cur_node == self.root:
                        cat = cur_node.cat
                    else:
                        par_k = int(samples[sample_idx, int(par_node.name)])
                        cat = cur_node.cat[par_k]

                    cur_sample = np.random.choice(np.arange(self.k), p=cat)
                    samples[sample_idx, int(cur_node.name)] = cur_sample
                    if len(cur_node.descendants) == 0:
                        filtered_samples[sample_idx, int(cur_node.name)] = cur_sample
                    else:
                        filtered_samples[sample_idx, int(cur_node.name)] = np.nan

        samples = samples.astype(int)
        self.samples = samples
        self.filtered_samples = filtered_samples
        self.num_samples = num_samples

    def get_topology_array(self):
        """ This function returns the tree topology as a numpy array. Each item represent the id of the parent node. """

        if self.num_leaves > 0:
            topology_array = np.zeros(self.num_nodes)

            visit_list = [self.root]
            while len(visit_list) != 0:
                cur_node = visit_list[0]
                visit_list = visit_list[1:]
                visit_list = visit_list + cur_node.descendants

                if cur_node.ancestor is None:
                    topology_array[int(cur_node.name)] = np.nan
                else:
                    topology_array[int(cur_node.name)] = cur_node.ancestor.name
        else:
            topology_array = np.array([])

        return topology_array

    def get_theta_array(self):
        """ This function returns the theta array. """

        theta_array = []
        for i in range(self.num_nodes):
            theta_array.append(np.zeros((self.k, self.k)))

        visit_list = [self.root]
        while len(visit_list) != 0:
            cur_node = visit_list[0]
            visit_list = visit_list[1:]
            visit_list = visit_list + cur_node.descendants
            theta_array[int(cur_node.name)] = cur_node.cat

        return theta_array

    def get_tree_newick(self):
        """ This function creates the Newick string of the tree. """

        sub_tree = tree_to_newick_rec(self.root)
        s = '[&R] (' + sub_tree + ')' + self.root.name + ';'
        return s

    def print_topology_array(self):
        """ This function prints the tree topology array. """

        print("Printing tree topology array... ")
        print("\t", self.get_topology_array())

    def print_topology(self):
        """ This function prints the tree topology with indentations. """

        if self.num_leaves > 0:
            print("Printing tree topology... ")

            visit_list = [self.root]
            visit_depth = [0]

            while len(visit_list) != 0:
                cur_node = visit_list[0]
                cur_depth = visit_depth[0]

                print("\t" * (cur_depth + 1) + cur_node.name)
                visit_list = visit_list[1:]
                visit_list = cur_node.descendants + visit_list
                visit_depth = visit_depth[1:]
                visit_depth = [cur_depth + 1] * len(cur_node.descendants) + visit_depth

    def print(self):
        """ This function prints all features of the tree. """

        if self.num_leaves > 0:
            print("Printing tree... ", self)
            print("\tnum_nodes: ", self.num_nodes, "\tnum_leaves: ", self.num_leaves, "\tk: ", self.k,
                  "\tnum_samples: ", self.num_samples, "\talpha: ", self.alpha, "\tNewick: ", self.newick)

            visit_list = [self.root]
            while len(visit_list) != 0:
                cur_node = visit_list[0]
                visit_list = visit_list[1:]
                cur_node.print()

                if len(cur_node.descendants) != 0:
                    visit_list = visit_list + cur_node.descendants

            if self.num_samples > 0:
                print("\tsamples:\n", self.samples)
                print("\tfiltered_samples:\n", self.filtered_samples)

    def save_tree(self, filename, save_arrays=False):
        """ This function saves the tree in a pickle file.
            If save_arrays=True, the function saves some of the features in numpy array format. """

        print("Saving tree to ", filename, "...")
        with open(filename, 'wb') as out_file:
            pickle.dump(self, out_file)

        newick_filename = filename + "_newick.txt"
        print("Saving Newick string to ", newick_filename, "...")
        with open(newick_filename, 'w') as out_file:
            out_file.write(self.newick)

        if save_arrays:
            topology_filename = filename + "_topology.npy"
            theta_filename = filename + "_theta.npy"
            samples_filename = filename + "_samples.npy"
            filtered_samples_filename = filename + "_filtered_samples.npy"
            print("Saving topology to ", topology_filename, ", theta to: ", theta_filename, ",  samples to ",
                  samples_filename, " and ", filtered_samples_filename, "...")
            np.save(topology_filename, self.get_topology_array())
            np.save(theta_filename, self.get_theta_array())
            np.save(samples_filename, self.samples)
            np.save(filtered_samples_filename, self.filtered_samples)

            topology_filename = filename + "_topology.txt"
            samples_filename = filename + "_samples.txt"
            filtered_samples_filename = filename + "_filtered_samples.txt"
            print("Saving topology to ", topology_filename, ",  samples to ", samples_filename, " and ",
                  filtered_samples_filename, "...")
            np.savetxt(topology_filename, self.get_topology_array(), delimiter="\t")
            np.savetxt(samples_filename, self.samples, delimiter="\t")
            np.savetxt(filtered_samples_filename, self.filtered_samples, delimiter="\t")

    def load_tree(self, filename):
        """ This function loads a tree from a pickle file. """

        print("Loading tree from ", filename, "...")
        with open(filename, 'rb') as in_file:
            t_temp = pickle.load(in_file)

        self.root = t_temp.root
        self.num_nodes = t_temp.num_nodes
        self.num_leaves = t_temp.num_leaves
        self.k = t_temp.k
        self.alpha = t_temp.alpha
        self.num_samples = t_temp.num_samples
        self.samples = t_temp.samples
        self.filtered_samples = t_temp.filtered_samples
        self.newick = t_temp.newick

    def load_tree_from_direct_arrays(self, topology_array, theta_array=[]):
        """ The 2019 version of the function is fixed by https://gits-15.sys.kth.se/alum.
            This function loads a tree directly from arrays.
            Example usage:
            topology_array = np.array([float('nan'), 0., 0.])
            theta_array = [
                np.array([0.5, 0.5]),
                np.array([[0.5, 0.5], [0.5, 0.5]]),
                np.array([[0.5, 0.5], [0.5, 0.5]])
            ]
            t = Tree()
            t.load_tree_from_direct_arrays(topology_array, theta_array)
        """

        print("Loading tree from topology_array...")
        k = 0

        self.root = Node(str(0), [])
        if len(theta_array) > 0:
            self.root.cat = theta_array[0]
            k = len(theta_array[0])

        visit_list = [self.root]

        num_nodes = 1
        num_leaves = 1
        while num_nodes < len(topology_array):
            cur_node = visit_list[0]
            visit_list = visit_list[1:]

            children_indices = np.where(topology_array == int(cur_node.name))[0]
            num_children = len(children_indices)

            if num_children > 0:
                num_leaves = num_leaves + num_children - 1
                children_list = []
                for child_idx in children_indices:
                    cat = []
                    if len(theta_array) > 0:
                        cat = theta_array[child_idx]

                    child_node = Node(str(child_idx), cat)
                    child_node.ancestor = cur_node
                    children_list.append(child_node)
                    visit_list.append(child_node)
                    num_nodes = num_nodes + 1
                cur_node.descendants = children_list

        self.num_nodes = num_nodes
        self.num_leaves = num_leaves
        self.k = k
        self.newick = self.get_tree_newick()

    def load_tree_from_arrays(self, topology_array_filename, theta_array_filename=None):
        """ This function loads a tree from numpy files. """

        print("Loading tree from topology_array: ", topology_array_filename,
              ", theta_array: ", theta_array_filename, "...")
        k = 0
        topology_array = np.load(topology_array_filename)
        if theta_array_filename is not None:
            theta_array = np.load(theta_array_filename, allow_pickle=True)
            k = len(theta_array[0])
        else:
            theta_array = []

        self.root = Node(str(0), [])
        if len(theta_array) > 0:
            self.root.cat = theta_array[0]

        visit_list = [self.root]

        num_nodes = 1
        num_leaves = 1
        while num_nodes < len(topology_array):
            cur_node = visit_list[0]
            visit_list = visit_list[1:]

            children_indices = np.where(topology_array == int(cur_node.name))[0]
            num_children = len(children_indices)

            if num_children > 0:
                num_leaves = num_leaves + num_children - 1
                children_list = []
                for child_idx in children_indices:
                    cat = []
                    if len(theta_array) > 0:
                        cat = theta_array[child_idx]

                    child_node = Node(str(child_idx), cat)
                    child_node.ancestor = cur_node
                    children_list.append(child_node)
                    visit_list.append(child_node)
                    num_nodes = num_nodes + 1
                cur_node.descendants = children_list

        self.num_nodes = num_nodes
        self.num_leaves = num_leaves
        self.k = k
        self.newick = self.get_tree_newick()


# Code taken from https://www.biostars.org/p/114387/
# Python program for Newick string generation, given a tree structure, provided by weslfield.
def tree_to_newick_rec(cur_node):
    """ This recursive function is a helper function to generate the Newick string of a tree. """

    items = []
    num_children = len(cur_node.descendants)
    for child_idx in range(num_children):
        s = ''
        sub_tree = tree_to_newick_rec(cur_node.descendants[child_idx])
        if sub_tree != '':
            s += '(' + sub_tree + ')'
        s += cur_node.descendants[child_idx].name
        items.append(s)
    return ','.join(items)


def main():
    print("Hello World!")
    print("This file demonstrates the usage of the functions.")

    print("\n1. Tree Generations\n")
    print("\n1.1. Create empty tree and print it:\n")
    t = Tree()
    t.print()

    print("\n1.2. Create a random tree and print it:\n")
    seed_val = 123
    k = 3
    t = Tree()
    t.create_random_tree(seed_val, k)
    t.print()

    print("\n1.3. Create a random tree with fixed number of nodes and print it:\n")
    num_nodes = 10
    t = Tree()
    t.create_random_tree_fix_nodes(seed_val, k, num_nodes=num_nodes, max_branch=3)
    t.print()

    print("\n1.4. Create a random binary tree and print it:\n")
    seed_val = 11
    k = 2
    num_nodes = 6
    t = Tree()
    t.create_random_binary_tree(seed_val, k, num_nodes=num_nodes)
    t.print()

    print("\n2. Sample Generation\n")
    print("\n2.1. Generate samples from tree and print it:\n")
    t.sample_tree(num_samples=5)
    t.print()

    print("\n3. Print Tree")
    print("\n3.1. Print all features of the tree:\n")
    t.print()

    print("\n3.2. Print the tree topology array:\n")
    t.print_topology_array()

    print("\n3.2. Print the tree topology in indentation form:\n")
    t.print_topology()

    print("\n4. Save Tree to file\n")
    filename = "data/examples/example_tree.pkl"
    t.save_tree(filename, save_arrays=True)

    print("\n5. Load Tree from file and print it:\n")
    print("\n5.1. Load tree from pickle file and print it:\n")
    t2 = Tree()
    t2.load_tree(filename)
    t2.print()

    print("\n5.2. Load tree from numpy arrays and print it:\n")
    topology_array = np.array([float('nan'), 0., 0.])
    theta_array = [
        np.array([0.5, 0.5]),
        np.array([[0.7, 0.3], [0.5, 0.5]]),
        np.array([[0.1, 0.9], [0.2, 0.8]])
    ]
    t2 = Tree()
    t2.load_tree_from_direct_arrays(topology_array, theta_array)
    t2.print()



if __name__ == "__main__":
    main()
