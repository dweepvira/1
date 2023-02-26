# Exercise-Pose-Estimation-and-Correction
Lifting weights is an essential component to any good exercise regime. However, it can also be dangerous if performed incorrectly. We present a solution that can detect and make suggestions to improve a user’s form in the main compound movements
Most weightlifting injuries occur as a result of using incorrect form while performing individual exercises. Incorrect form has become an increasingly problematic issue in the wake of COVID. However, without feedback from a personal trainer and minimal prior experience, it can be quite a difficult task to learn how to lift weights properly. The proliferation of machine learning research in the past decade has also trickled into exercise science
The approach we present provides a very promising initial solution to the issue of incorrect form while performing exercises. We approached our solution through 3 parts:
1.) classifying an exercise,
2.) generating the human pose model for the image, and
3.) generating feedback so that the user can improve their form.
Because we treated each of these 3 parts as discrete stages with each stage outputting the input to the next, we were able to calibrate each stage individually so that we were able to iteratively improve upon the solution.
