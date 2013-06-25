%%%-------------------------------------------------------------------
%%% @author cantheman <java10cana@gmail.com>
%%% @copyright (C) 2013, cantheman
%%% @doc
%%%
%%% @end
%%% Created : 16 Mar 2013 by cantheman <java10cana@gmail.com>
%%%-------------------------------------------------------------------
-module(e_ann_output_neuron).

-behaviour(gen_server).

%% API
-export([start_link/1, add_input/2]).

%% gen_server callbacks
-export([init/1, handle_call/3, handle_cast/2, handle_info/2,
	 terminate/2, code_change/3]).

-define(SERVER, ?MODULE).

-record(state, {global_error=0.0, ideal_output=0.0, input_list=[]}).

%%%===================================================================
%%% API
%%%===================================================================
start_link(Args) ->
    gen_server:start_link(?MODULE, [Args], []).

add_input(NeuronPid, Input) ->
    gen_server:call(NeuronPid, {add_to_input_list, Input}).

%%%===================================================================
%%% gen_server callbacks
%%%===================================================================

init([Ideal]) ->
    log4erl:log(info, "Starting (~p) output neuron ideal output of:~p~n",
                [self(), Ideal]),
    State = #state{ideal_output=Ideal},
    {ok, State}.

handle_call({add_to_input_list, Input}, _From, State) ->
    InputList = State#state.input_list,
    NewInputList = [Input | InputList],
    log4erl:log(info, "(~p) added ~p to input_list~n",[self(), Input]),
    NewState = #state{input_list=NewInputList},
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
