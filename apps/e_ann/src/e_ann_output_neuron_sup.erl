%%%-------------------------------------------------------------------
%%% @author cantheman <cantheman@campanja.com>
%%% @copyright (C) 2013, cantheman
%%% @doc
%%%
%%% @end
%%% Created : 17 Mar 2013 by cantheman <cantheman@campanja.com>
%%%-------------------------------------------------------------------
-module(e_ann_output_neuron_sup).

-behaviour(supervisor).

%% API
-export([start_link/0, start_child/0]).

%% Supervisor callbacks
-export([init/1]).

-define(SERVER, ?MODULE).

%%%===================================================================
%%% API functions
%%%===================================================================
start_link() ->
    supervisor:start_link({local, ?SERVER}, ?MODULE, []).

%%%===================================================================
%%% Supervisor callbacks
%%%===================================================================
init(_Args) ->
    log4erl:log(info, "Starting output_neuron_supervisor (~p) ~n", [self()]),
    RestartStrategy = {one_for_one, 5, 10},
    {ok, {RestartStrategy, []}}.

start_child() ->
    supervisor:start_child(?MODULE, []).

%%%===================================================================
%%% Internal functions
%%%===================================================================
%% child(Module) ->
%%     {Module, {Module, start_link, []}, temporary, 2000, worker, [Module]}.
