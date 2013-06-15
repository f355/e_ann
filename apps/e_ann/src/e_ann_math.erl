-module(e_ann_math).

-export([mse/1, ess/1, rms/1, sigmoid/1,
         output_delta/2, linear_error/2]).
-compile([export_all]).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Global Error Functions %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Mean Squared Error
mse(Errors) ->
    Errs = [ squared_diff(X) || X <- Errors ],
    lists:sum(Errs) / length(Errors).

%% Sum of Squares Error
ess(Errors) ->
    Errs = [ squared_diff(X) || X <- Errors ],
    lists:sum(Errs) / 0.5.

%% Root Mean Square Error
rms(Errors) ->
    Errs = [ squared_diff(X) || X <- Errors ],
    Sum = lists:sum(Errs) / length(Errors),
    math:sqrt(Sum).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Activation Functions %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Sigmoid Function
sigmoid(Number) ->
    1 / (1 + (math:exp(-Number))).

linear_error(Ideal, Actual) ->
    Actual - Ideal.

squared_diff(Error) ->
    math:pow(Error, 2).

derivative_sigmoid(Sum) ->
    sigmoid(Sum) * (1.0 - sigmoid(Sum)).

output_delta(E, Sum) ->
    -E * derivative_sigmoid(Sum).

interior_delta(Sum, Delta, Weight) ->
    derivative_sigmoid(Sum) * (Delta * Weight).
