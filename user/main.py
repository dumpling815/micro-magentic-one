import os, time, httpx
from RequestSchema import InvokeBody, InvokeResult, Msg

REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "120"))
PORT = os.getenv("PORT", "8000")
RETRIES = int(os.getenv("RETRIES", "1"))             # 실패 시 추가 재시도 횟수

def agent_health_check():
    # Check if all agents are healthy
    agents = ["filesurfer", "websurfer", "coder", "computerterminal", "orchestrator"]  
    try:
        with httpx.Client(timeout=5.0) as client:
            for agent in agents:
                response = client.get(f"http://{agent}:8000/health")
                if response.status_code != 200:
                    print(f"Agent {agent} health check failed: {response.text}")
                    return False
        return True
    except httpx.HTTPError as e:
        print(f"Error during health check: {e}")
        return False

def main():
    setup_trial = 0
    while True:
        try:
            if agent_health_check():
                print("All agents are ready.")
                break
            print("One or more agents are not ready. Retrying...")
            setup_trial += 1
            if setup_trial == 6:
                print("Agents are not responding after multiple attempts. Exiting.")
                return
            time.sleep(5)
        except Exception as e:
            print(f"Error during agent health check: {e}")
            setup_trial += 1
            time.sleep(5)
    
    # Start the main loop if all agents are ready
    with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
        time_per_request_ms = []
        while True:
            try:
                user_input = input("Your request: ")
                if user_input.lower() in ["exit", "quit", "q", "bye", "quit()", "exit()"]:
                    print("Terminating Agent System...")
                    # Clean up code might be implemented here if needed (Docker Container 정리)
                    # 필요시 Docker Container에게 /kill url로 요청
                    break
                elif user_input == "":
                    continue

                start_time_perf = time.perf_counter()
                user_msg = Msg(type="TextMessage", source="user", content=user_input)
                final_response = client.post(f"http://localhost:{PORT}/start", json=user_msg.model_dump())
                if final_response.status_code in [200, 201, 204, 206]:
                    result = InvokeResult(**final_response.json())
                    time_per_request_ms.append(int((time.perf_counter() - start_time_perf)*1000))
                    print("####################################")
                    print(f"Status:{result.status}\nSteps:{result.steps}\nTotal Latency(ms):{result.total_latency_ms}\n")
                    print("####################################")
                    print(f"Content: {result.message.content}")
                    print("####################################")
                else:
                    time_per_request_ms.append(int((time.perf_counter() - start_time_perf)*1000))
                    print(f"Error: {final_response.text}")
            except httpx.RequestError as e:
                print(f"Request failed: {e}")
            except (KeyboardInterrupt, EOFError):
                print("Terminating Agent System...")
                break

if __name__ == "__main__":
    main()