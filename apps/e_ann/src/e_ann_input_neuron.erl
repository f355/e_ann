%%%-------------------------------------------------------------------
%%% @author cantheman <java10cana@gmail.com>
%%% @copyright (C) 2013, cantheman
%%% @doc
%%% This module spawns input neurons that receive an input upon init.
%%% Random weights are then assigned and feedforward commences.
%%% @end
%%% Created :  9 Mar 2013 by cantheman <java10cana@gmail.com>
%%%-------------------------------------------------------------------
-module(e_ann_input_neuron).

-behaviour(gen_server).

%% API
-export([start_link/1, feed_forward/2, init_weights/2,
         forward_output/2]).

%% gen_server callbacks
-export([init/1, handle_call/3, handle_cast/2, handle_info/2,
	 terminate/2, code_change/3]).

-define(SERVER, ?MODULE).

-record(state, {input=0.0,
                weights=[],
                feedforward_values=[]}).

%%%===================================================================
%%% API
%%%===================================================================
start_link(Args) ->
    gen_server:start_link(?MODULE, [Args], []).

feed_forward(NeuronPid, TargetPids) ->
    gen_server:call(NeuronPid, {feed_forward, TargetPids}).

init_weights(NeuronPid, Count) ->
    gen_server:call(NeuronPid, {init_weights, Count}).

%%%===================================================================
%%% gen_server callbacks
%%%===================================================================

init([Input]) ->
    log4erl:log(info, "Starting ~p Input neuron with input:~p ~n",
		[self(), Input]),
    State = #state{input=Input},
    {ok, State}.

handle_call({init_weights, Count}, _From, State) ->
    Weights = e_ann_math:generate_random_weights(Count),
    NewState = State#state{weights=Weights},
    log4erl:log(info, "(~p) initialized with weights ~p~n",[self(), Weights]),
    {reply, ok, NewState};
handle_call({feed_forward, TargetPids}, _From, State) ->
    Input = State#state.input,
    Weights = State#state.weights,
    FeedForwardValues = [ Input * Weight || Weight <- Weights ],
    NewState = State#state{feedforward_values=FeedForwardValues},
    forward_output(FeedForwardValues, TargetPids),
    {reply, ok, NewState};
handle_call(_Request, _From, State) ->
    {reply, ok, State}.

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

forward_output([], []) ->
    ok;
forward_output(Values, TargetNeurons) ->
    e_ann_hidden_neuron:add_input(hd(TargetNeurons), hd(Values)),
    forward_output(tl(Values), tl(TargetNeurons)).
