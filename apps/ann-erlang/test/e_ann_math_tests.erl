-module(e_ann_math_tests).

-include_lib("eunit/include/eunit.hrl").

mse_test() ->
    Errs = get_error_data(),
    Result = e_ann_math:mse(Errs),
    ?assertEqual(0.051186, Result).

get_error_data() ->
    [0.23, 0.05, -0.039, -0.42, 0.32, -0.15, 0.009].
