#       We haven't told RANSAC algorithm this week. So please try to do the reading.
#       And now, we can describe it here:
#       We have 2 sets of points, say, Points A and Points B. We use A.1 to denote the first point in A,
#       B.2 the 2nd point in B and so forth. Ideally, A.1 is corresponding to B.1, ... A.m corresponding
#       B.m. However, it's obvious that the matching cannot be so perfect and the matching in our real
#       world is like:
#       A.1-B.13, A.2-B.24, A.3-x (has no matching), x-B.5, A.4-B.24(This is a wrong matching) ...
#       The target of RANSAC is to find out the true matching within this messy.
#
#       Algorithm for this procedure can be described like this:
#       1. Choose 4 pair of points randomly in our matching points. Those four called "inlier" (中文： 内点) while
#          others "outlier" (中文： 外点)
#       2. Get the homography of the inliers
#       3. Use this computed homography to test all the other outliers. And separated them by using a threshold
#          into two parts:
#          a. new inliers which is satisfied our computed homography
#          b. new outliers which is not satisfied by our computed homography.
#       4. Get our all inliers (new inliers + old inliers) and goto step 2
#       5. As long as there's no changes or we have already repeated step 2-4 k, a number actually can be computed,
#          times, we jump out of the recursion. The final homography matrix will be the one that we want.
#
#       [WARNING!!! RANSAC is a general method. Here we add our matching background to that.]
#
#       Your task: please complete pseudo code (it would be great if you hand in real code!) of this procedure.
#
#       Python:
#       def ransacMatching(A, B):
#           A & B: List of List
#
import numpy as np
import cv2

def random_sample_sets(A, B):

    pts = np.random.randint(A.shape[0], size=4)

    match_set = [(A[i,:], B[i,:]) for i in pts]

    return match_set


def compute_homography(match_set):
    pass


def is_inlier(ptA, ptB, homography, threshold):

    return True

def ransacMatching(A, B):

    matrix_A = np.matrix(A)
    matrix_B = np.matrix(B)

    match_threshold = 0.6

    max_iter = 2000

    inlier_threshold = 0

    homography = None

    match_score = 0

    for _ in range(max_iter):
        # Repeat steps to maximize match

        inlier = []

        # Step 1: Randomly sample sets of 4 point matches
        match_set = random_sample_sets(matrix_A, matrix_B)


        # Step 2: Compute homography of the inliers
        h = compute_homography(match_set)

        # Step 3: Use this computed homography to test all the other outliers. And separated them by using a threshold
        for i in range(matrix_A.shape[0]):
            if is_inlier(matrix_A[i,:], matrix_B[i,:], h, inlier_threshold):
                inlier.append(i)

        if len(inlier) >=  match_threshold * matrix_A.shape[0]:
            break

        homography = homography if match_score > len(inlier) else h

        match_score = len(inlier) if match_score < len(inlier) else match_score

    return homography





