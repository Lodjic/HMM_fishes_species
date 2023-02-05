from player_controller_hmm import PlayerControllerHMMAbstract
import numpy as np
import time
import math


##### Model contruction thanks to the inputs #####

def construct_almost_uniform_matrix(nb_rows, nb_columns, espilon=0.03):
    M = []
    for i in range(nb_rows):
        M.append([1/nb_columns for j in range(nb_columns)])
    for i in range(nb_rows):
        for j in range(nb_columns):
            M[i][j] += np.random.random() * (2 * espilon) - espilon
        row_sum = sum(M[i])
        for j in range(nb_columns):
            M[i][j] /= row_sum
    return M


##### Useful functions on lists #####

def where_equal(l, condition):
    indexes = []
    for i in range(len(l)):
        if l[i] == condition:
            indexes.append(i)
    return indexes

def where_not_equal(l, condition):
    indexes = []
    for i in range(len(l)):
        if l[i] != condition:
            indexes.append(i)
    return indexes


##### HMM useful fcts #####

def compute_alpha_scaled_coeffs(A, B, pi, obs):
    T = len(obs) - 1  # index of final step
    nb_states = len(A)
    alpha_mat = [[0] * (T+1) for k in range(nb_states)]
    scaling_vec = [0] * (T+1)
    # alpha 0
    c0 = 0
    for i in range(nb_states):
        alpha_val = pi[0][i] * B[i][obs[0]]
        alpha_mat[i][0] = alpha_val
        c0 += alpha_val
    c0 = 1/c0
    scaling_vec[0] = c0
    for i in range(nb_states):
        alpha_mat[i][0] *= c0
    # alpha_t
    for t in range(1, T+1):
        c = 0
        for i in range(nb_states):
            alpha_val = B[i][obs[t]] * sum([alpha_mat[j][t-1] * A[j][i] for j in range(nb_states)])
            alpha_mat[i][t] = alpha_val
            c += alpha_val
        c = 1/c
        scaling_vec[t] = c
        for i in range(nb_states):
            alpha_mat[i][t] *= c
    return alpha_mat, scaling_vec

def compute_beta_scaled_coeffs(A, B, obs, scaling_vec):
    T = len(obs) - 1  # index of final step
    nb_states = len(A)
    beta_mat = [[0] * (T+1) for k in range(nb_states)]
    # beta T
    for i in range(nb_states):
        beta_mat[i][T] = scaling_vec[T]
    # beta t
    for t in range(T-1, -1, -1):
        for i in range(nb_states):
            beta_mat[i][t] = scaling_vec[t] * sum([beta_mat[j][t+1] * A[i][j] * B[j][obs[t+1]] for j in range(nb_states)])
    return beta_mat

def di_gamma(t, i, j, A, B, obs, alpha_mat, beta_mat):
    return (alpha_mat[i][t] * A[i][j] * B[j][obs[t+1]] * beta_mat[j][t+1])

def compute_di_gamma_coeffs(A, B, obs, alpha_mat, beta_mat):
    T = len(obs) - 1  # index of final step
    nb_states = len(A)
    di_gamma_mat = [[[0] * nb_states for k in range(T)] for h in range(nb_states)]
    for t in range(T):  # go until (T-1)
        for i in range(nb_states):
            for j in range(nb_states):
                di_gamma_mat[i][t][j] = di_gamma(t, i, j, A, B, obs, alpha_mat, beta_mat)
    return di_gamma_mat

def compute_gamma_coeffs(A, obs, di_gamma_mat, alpha_mat):
    T = len(obs) - 1  # index of final step
    nb_states = len(A)
    gamma_mat = [[0] * T for k in range(nb_states)]
    for t in range(T-1):  # go until (T-1)
        for i in range(nb_states):
            gamma_mat[i][t] = sum([di_gamma_mat[i][t][j] for j in range(nb_states)])
    for i in range(nb_states):
        gamma_mat[i][T-1] = alpha_mat[i][T-1]
    return gamma_mat


####################### Fcts to update matrixes ####################### 

def update_A(A, di_gamma_mat, gamma_mat):
    new_A = [[-1] * len(A[0]) for k in range(len(A))]
    for i in range(len(A)):
        for j in range(len(A[0])):
            new_A[i][j] = sum([di_gamma_mat[i][t][j] for t in range(len(di_gamma_mat[0]))]) / sum([gamma_mat[i][t] for t in range(len(di_gamma_mat[0]))])
    return new_A

def update_B(B, obs, gamma_mat):
    new_B = [[-1] * len(B[0]) for k in range(len(B))]
    for j in range(len(B)):
        for k in range(len(B[0])):
            b_j_k = 0
            for t in range(len(gamma_mat[0])):
                if obs[t] == k:
                    b_j_k += gamma_mat[j][t]
            new_B[j][k] = b_j_k / sum([gamma_mat[j][t_step] for t_step in range(len(gamma_mat[0]))])
    return new_B

def update_pi(pi, gamma_mat):
    new_pi = [[-1] * len(pi[0])]
    for i in range(len(pi[0])):
        new_pi[0][i] = gamma_mat[i][0]
    return new_pi

def autoupdate_A(A, di_gamma_mat, gamma_mat):
    for i in range(len(A)):
        for j in range(len(A[0])):
            A[i][j] = sum([di_gamma_mat[i][t][j] for t in range(len(di_gamma_mat[0]))]) / sum([gamma_mat[i][t] for t in range(len(di_gamma_mat[0]))])
    return

def autoupdate_B(B, obs, gamma_mat):
    for j in range(len(B)):
        for k in range(len(B[0])):
            b_j_k = 0
            for t in range(len(gamma_mat[0])):
                if obs[t] == k:
                    b_j_k += gamma_mat[j][t]
            B[j][k] = b_j_k / sum([gamma_mat[j][t_step] for t_step in range(len(gamma_mat[0]))])
    return

def autoupdate_pi(pi, gamma_mat):
    for i in range(len(pi[0])):
        pi[0][i] = gamma_mat[i][0]
    return

def compute_log_proba(scaling_vec):
    return -sum([math.log10(c) for c in scaling_vec])

def calculate_log_proba_of_obs_seq(A, B , pi, obs):
    try :
        alpha_mat, scaling_vec = compute_alpha_scaled_coeffs(A, B, pi, obs)
        return -sum([math.log10(c) for c in scaling_vec])
    except ZeroDivisionError:
        return -10000000

def compute_mat_diff(mat1, mat2):
    assert len(mat1) == len(mat2) and len(mat1[0]) == len(mat2[0]), "Error the 2 matrices are not the same length"
    diff = 0
    for i in range(len(mat1)):
        for j in range(len(mat1[0])):
            diff += (mat1[i][j] - mat2[i][j]) ** 2
    return diff


####################### Baum-Welch algorithm #######################

def baum_welch_one_iteration(A, B, pi, obs):
    alpha_mat, scaling_vec = compute_alpha_scaled_coeffs(A, B, pi, obs)
    beta_mat = compute_beta_scaled_coeffs(A, B, obs, scaling_vec)
    di_gamma_mat = compute_di_gamma_coeffs(A, B, obs, alpha_mat, beta_mat)
    gamma_mat = compute_gamma_coeffs(A, obs, di_gamma_mat, alpha_mat)
    # updating matrixes
    new_A = update_A(A, di_gamma_mat, gamma_mat)
    new_B = update_B(B, obs, gamma_mat)
    new_pi = update_pi(pi, gamma_mat)
    A, B, pi = new_A, new_B, new_pi
    return

def baum_welch(A, B, pi, obs, time_max):
    t0 = time.time()
    old_log_proba = -100000000
    log_proba = -10000000
    iter = 0
    while time.time() - t0 < time_max and log_proba > old_log_proba and iter < 25: 
        iter += 1
        alpha_mat, scaling_vec = compute_alpha_scaled_coeffs(A, B, pi, obs)
        beta_mat = compute_beta_scaled_coeffs(A, B, obs, scaling_vec)
        di_gamma_mat = compute_di_gamma_coeffs(A, B, obs, alpha_mat, beta_mat)
        gamma_mat = compute_gamma_coeffs(A, obs, di_gamma_mat, alpha_mat)
        # updating matrixes
        autoupdate_A(A, di_gamma_mat, gamma_mat)
        autoupdate_B(B, obs, gamma_mat)
        autoupdate_pi(pi, gamma_mat)
        # calculationg and updating new log_proba
        old_log_proba = log_proba
        log_proba = compute_log_proba(scaling_vec)
    return


####################### Player controller class #######################
class PlayerControllerHMM(PlayerControllerHMMAbstract):
    def init_parameters(self):
        """
        In this function you should initialize the parameters you will need,
        such as the initialization of models, or fishes, among others.
        """
        nb_states = 4
        nb_species = 7
        nb_fishes = 70

        self.A_matrixes = []
        for _ in range(nb_species):
            self.A_matrixes.append(construct_almost_uniform_matrix(nb_states, nb_states, 0.05))
        self.B_matrixes = []
        for _ in range(nb_species):
            self.B_matrixes.append(construct_almost_uniform_matrix(nb_states, 8, 0.02))
        self.pi_matrixes = []
        for _ in range(nb_species):
            self.pi_matrixes.append(construct_almost_uniform_matrix(1, nb_states, 0.05))
        self.observations = [[] for i in range(nb_fishes)]
        self.fish_species = [-1] * nb_fishes

        return

    def guess(self, step, observations):
        """
        This method gets called on every iteration, providing observations.
        Here the player should process and store this information,
        and optionally make a guess by returning a tuple containing the fish index and the guess.
        :param step: iteration number
        :param observations: a list of N_FISH observations, encoded as integers
        :return: None or a tuple (fish_id, fish_type)
        """

        nb_states = 4
        nb_species = 7
        nb_fishes = 70

        for i in range(nb_fishes):
            self.observations[i].append(observations[i])
        
        if step >= 90:
            index_fish_to_guess = np.random.choice(where_equal(self.fish_species, -1))
            proba_of_obs_seq = []
            #print(f"Fish id guessed : {index_fish_to_guess}")
            for i in range(nb_species):
                proba_of_obs_seq.append(calculate_log_proba_of_obs_seq(self.A_matrixes[i], self.B_matrixes[i] , self.pi_matrixes[i], self.observations[index_fish_to_guess]))
            #print(proba_of_obs_seq)
            return (index_fish_to_guess, proba_of_obs_seq.index(max(proba_of_obs_seq)))
        else:
            return None


    def reveal(self, correct, fish_id, true_type):
        """
        This methods gets called whenever a guess was made.
        It informs the player about the guess result
        and reveals the correct type of that fish.
        :param correct: tells if the guess was correct
        :param fish_id: fish's index
        :param true_type: the correct type of the fish
        :return:
        """
        nb_states = 4
        nb_species = 7
        nb_fishes = 70

        self.fish_species[fish_id] = true_type
        #print(f"True type: {true_type}")
        if correct is True:
            return
        else:
            try:
                fish_already_known_indexes = where_equal(self.fish_species, true_type)
                time_max = 1/len(fish_already_known_indexes)
                for fish_index in fish_already_known_indexes:
                    #print(f"Baum-welch for fish index {fish_index} of species {self.fish_species[fish_index]}")
                    #print(self.A_matrixes[true_type])
                    baum_welch(self.A_matrixes[true_type], self.B_matrixes[true_type], self.pi_matrixes[true_type], self.observations[fish_index], time_max)
                return
                
            except ZeroDivisionError:
                #print(f"Exception raised of type {true_type}")
                self.A_matrixes[true_type] = construct_almost_uniform_matrix(nb_states, nb_states, 0.05)
                self.B_matrixes[true_type] = construct_almost_uniform_matrix(nb_states, 8, 0.02)
                self.pi_matrixes[true_type] = construct_almost_uniform_matrix(1, nb_states, 0.05)
                baum_welch(self.A_matrixes[true_type], self.B_matrixes[true_type], self.pi_matrixes[true_type], self.observations[fish_id], 1)
                return
