-module(e_ann_math).

-export([mse/1, ess/1, rms/1, sigmoid/1,
         output_delta/2, linear_error/2,
         hyperbolic_tangent/1, activation/1]).
-compile([export_all]).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Global Error Functions %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% @doc Mean Squared Error
mse(Errors) ->
    Errs = [ squared_diff(X) || X <- Errors ],
    lists:sum(Errs) / length(Errors).

%% @doc Sum of Squares Error
ess(Errors) ->
    Errs = [ squared_diff(X) || X <- Errors ],
    lists:sum(Errs) / 0.5.

%% @doc Root Mean Square Error
rms(Errors) ->
    Errs = [ squared_diff(X) || X <- Errors ],
    Sum = lists:sum(Errs) / length(Errors),
    math:sqrt(Sum).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Activation Functions %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% @doc Sigmoid Function
sigmoid(N) ->
    1 / (1 + (math:exp(-N))).

%% @doc Hyperbolic Tangent Function
hyperbolic_tangent(N) ->
    (math:exp(2*N) - 1) / (math:exp(2*N) + 1).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Misc Functions %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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

activation(Inputs) ->
    Sum = lists:sum(Inputs),
    e_ann_math:sigmoid(Sum).

generate_random_weight() ->
    {_, _, Random} = random:seed(now()),
    Float = list:concat(["0", ".", integer_to_list(Random)]),
    list_to_float(Float).
