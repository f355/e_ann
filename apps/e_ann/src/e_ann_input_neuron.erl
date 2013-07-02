%%%-------------------------------------------------------------------
%%% @author cantheman <java10cana@gmail.com>
%%% @copyright (C) 2013, cantheman
%%% @doc
%%%
%%% @end
%%% Created :  9 Mar 2013 by cantheman <java10cana@gmail.com>
%%%-------------------------------------------------------------------
-module(e_ann_input_neuron).

-behaviour(gen_server).

%% API
-export([start_link/1, calculate_output/2, init_weights/2]).

%% gen_server callbacks
-export([init/1, handle_call/3, handle_cast/2, handle_info/2,
	 terminate/2, code_change/3]).

-define(SERVER, ?MODULE).

-record(state, {input=0.0, weights=[], outputs=[]}).

%%%===================================================================
%%% API
%%%===================================================================
start_link(Args) ->
    gen_server:start_link(?MODULE, [Args], []).

calculate_output(NeuronPid, TargetPids) ->
    gen_server:call(NeuronPid, {calculate_output, TargetPids}).

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
handle_call({calculate_output, TargetPids}, _From, State) ->
    Input = State#state.input,
    Weights = State#state.weights,
    Outputs = [ Input * Weight || Weight <- Weights ],
    NewState = State#state{outputs=Outputs},
    [ e_ann_hidden_neuron:add_input(Pid, Output) || Pid <- TargetPids,
                                                    Output <- Outputs ],
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

