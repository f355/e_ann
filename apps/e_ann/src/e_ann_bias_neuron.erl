%%%-------------------------------------------------------------------
%%% @author cantheman <java10cana@gmail.com>
%%% @copyright (C) 2013, cantheman
%%% @doc
%%% Bias neuron with static input of 1.
%%% @end
%%% Created : 10 Mar 2013 by cantheman <java10cana@gmail.com>
%%%-------------------------------------------------------------------
-module(e_ann_bias_neuron).

-behaviour(gen_server).

%% API
-export([start_link/0, calculate_output/2]).

%% gen_server callbacks
-export([init/1, handle_call/3, handle_cast/2, handle_info/2,
	 terminate/2, code_change/3]).

-define(SERVER, ?MODULE).
-define(INPUT, 1).

-record(state, {weight=0.0, output=0.0}).


%%%===================================================================
%%% API
%%%===================================================================
start_link() ->
    gen_server:start_link(?MODULE, [], []).

calculate_output(NeuronPid, TargetPids) ->
    gen_server:call(NeuronPid, {calculate_output, TargetPids}).

%%%===================================================================
%%% gen_server callbacks
%%%===================================================================

init([]) ->
    Weight = e_ann_math:generate_random_weight(),
    log4erl:log(info, "Starting ~p Bias neuron with weight of ~p~n",
		[self(),Weight]),
    State = #state{weight=Weight},
    {ok, State}.

handle_call({calculate_output, TargetPids}, _From, State) ->
    Weight = State#state.weight,
    Output = ?INPUT * Weight,
    NewState = State#state{output=Output},
    [ e_ann_hidden_neuron:add_input(Pid, Output) || Pid <- TargetPids ],
    {reply, ok, NewState};
handle_call(_Request, _From, State) ->
    Reply = ok,
    {reply, Reply, State}.

handle_cast(_Msg, State) ->
    {noreply, State}.

handle_info(_Info, State) ->
    {noreply, State}.

terminate(_Reason, _State) ->
    ok.

code_change(_OldVsn, State, _Extra) ->
    {ok, State}.

%%%===================================================================
%%% Internal functions
%%%===================================================================

