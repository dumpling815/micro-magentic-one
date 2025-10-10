import os, time, httpx
from common.request_schema import InvokeBody, InvokeResult, Msg
from autogen_agentchat.base import TaskResult

REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT"))
RETRIES = int(os.getenv("RETRIES"))             # 실패 시 추가 재시도 횟수


def agent_health_check():
    # Check if all agents are healthy
    # TODO: health_check는 orchestrator에 둬야하는게 맞지 않나. -> 점검 필요.
    HEALTH_PATH = os.getenv("HEALTH_PATH")
    agents = ["filesurfer", "websurfer", "coder", "orchestrator"]
    agent_urls = [os.getenv(agent.upper() + "_URL") + HEALTH_PATH for agent in agents]
    try:
        with httpx.Client(timeout=5.0) as client:
            for agent_url in agent_urls:
                response = client.get(agent_url)
                print(f"Agent {agent_url} response : {response}")
                if response.status_code != 200:
                    print(f"Agent {agent_url} health check failed: {response.text}")
                    return False
    except httpx.HTTPError as e:
        print(f"Error during health check: {e}")
    try:
        with httpx.Client(timeout=5.0) as client:
            response = client.get("http://localhost:8080/health")
            if response.status_code != 200:
                print(f"Computerterminal health check failed: {response.text}")
                return False
    except httpx.HTTPError as e:
        print(f"Error during health check(computer terminal): {e}")
        return False
    return True

def main():
    setup_trial = 0
    while True:
        try:
            if agent_health_check():
                print("All agents are ready.")
                break
            print("One or more agents are not ready. Retrying...")
            setup_trial += 1
            if setup_trial == RETRIES:
                print("Agents are not responding after multiple attempts. Exiting.")
                return
            time.sleep(5)
        except Exception as e:
            print(f"Error during agent health check: {e}")
            setup_trial += 1
            time.sleep(5)
    
    # Start the main loop if all agents are ready
    ORCHESTRATOR_URL = os.getenv("ORCHESTRATOR_URL")
    with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
        e2e_time_per_request_ms = []
        orchestrate_time_per_request_ms = []
        while True:
            try:
                user_input = input("Your request: ")
                if user_input.lower() in ["exit", "quit", "q", "bye", "quit()", "exit()"]:
                    print("Terminating Agent System...")
                    # Clean up code might be implemented here if needed (Docker Container 정리)
                    # 필요시 Docker Container에게 /kill url로 요청
                    break
                elif user_input == "":
                    print("Task cannot be empty. Please enter a valid request.")
                    break

                # Orchestration Request
                start_time_perf = time.perf_counter()
                user_msg = Msg(type="TextMessage", source="user", content=user_input)
                body = InvokeBody(messages=[user_msg])
                final_response: InvokeResult = client.post(ORCHESTRATOR_URL + "/invoke", json=body.model_dump()) # Httpx를 통해 요청할 때 Json으로 직렬화 필요.
            
            except httpx.RequestError as e:
                print(f"Request failed: {e}")
            except (KeyboardInterrupt, EOFError):
                print("Terminating Agent System...")
                break

            # Deserialization
            result = InvokeResult(**final_response) # 응답 형태는 Json, 역직렬화.
            task_result: TaskResult = result.response if isinstance(result.response, TaskResult) else None 

            # Latency Measurement
            end_time_perf = time.perf_counter()
            e2e_time_per_request_ms.append(int((end_time_perf - start_time_perf)*1000))
            orchestrate_time_per_request_ms.append(result.elapsed.get("orchestration_latency_ms", 0))
            # Display Result of the Entire System
            print("####################################")
            print(f"Final Message from Micro Magentic-One System:\n->  {task_result.messages[-1].content if task_result and task_result.messages else 'No message returned'}")
            print("####################################")
            print(f"E2E Latency(ms): {e2e_time_per_request_ms[-1]}\nOrchestration Latency(ms): {orchestrate_time_per_request_ms[-1]}")
            print("####################################")
            print(f"Enter 'y' if you want to see the full conversation history, anything else to continue.",end="")
            if input().lower() == 'y':
                if task_result and task_result.messages:
                    print("Full Conversation History:")
                    for msg in task_result.messages:
                        print(f"[{msg.source}]: {msg.content}")
                else:
                    print("No conversation history available.")

if __name__ == "__main__":
    main()