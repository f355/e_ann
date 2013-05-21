-module(e_ann_math).

-export([mse/1, ess/1, rms/1, sigmoid/1]).

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

%% Sigmoid Function
sigmoid(Number) ->
    1 / (1 + (math:exp(-Number))).

squared_diff(Error) ->
    math:pow(Error, 2).

