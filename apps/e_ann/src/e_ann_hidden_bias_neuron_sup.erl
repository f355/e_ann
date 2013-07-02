%%%-------------------------------------------------------------------
%%% @author cantheman <java10cana@gmail.com>
%%% @copyright (C) 2013, cantheman
%%% @doc
%%% Bias neuron supervisor who dynamically spawns bias neurons.
%%% @end
%%% Created : 13 June 2013 by cantheman <java10cana@gmail.com>
%%%-------------------------------------------------------------------
-module(e_ann_hidden_bias_neuron_sup).

-behaviour(supervisor).

%% API
-export([start_link/0, add_child/1]).

%% Supervisor callbacks
-export([init/1]).

%% ===================================================================
%% API functions
%% ===================================================================

start_link() ->
    supervisor:start_link(?MODULE, []).

%% ===================================================================
%% Supervisor callbacks
%% ===================================================================

init([]) ->
    log4erl:log(info, "Starting e_ann_bias_neuron supervisor (~p)~n",
                [self()]),
    RestartStrategy = {simple_one_for_one, 0, 1},
    Children = [child(e_ann_hidden_bias_neuron)],
    {ok, {RestartStrategy, Children}}.

add_child(Sup) ->
    supervisor:start_child(Sup, []).

child(Module) ->
    {Module, {Module, start_link, []}, temporary, 2000, worker, [Module]}.
