# ──────────────────────────────────────────────────────────────────────────────
# Core/sql.py
# MySQL 데이터베이스 연결 및 쿼리 실행을 담당하는 모듈입니다.
# 설정 로드 우선순위: 환경변수 > config.json > server_config.py
# ──────────────────────────────────────────────────────────────────────────────
import os
import json
import mysql.connector
from mysql.connector import Error
import Core.server_config as server_config


def _load_mysql_config():
    """
 MySQL 연결 설정을 로드합니다.

 우선순위:
   1) 환경변수 MYSQL_* (최우선)
   2) config.json (server_config.DEFAULT_CFG 경로)
   3) Core.server_config.SQL (최종 fallback)
 """
    # 1) 환경변수 확인
    cfg_env = {
        "host": os.getenv("MYSQL_HOST"),
        "database": os.getenv("MYSQL_DATABASE"),
        "user": os.getenv("MYSQL_USER"),
        "password": os.getenv("MYSQL_PASSWORD"),
        "port": os.getenv("MYSQL_PORT"),
    }

    def _is_complete(d: dict) -> bool:
        """필수 설정값이 모두 존재하는지 확인"""
        return all(d.get(k) not in (None, "",) for k in ("host", "database", "user", "password"))

    if _is_complete(cfg_env):
        cfg_env["port"] = int(cfg_env["port"] or 3306)
        return cfg_env

    # 2) config.json 파일 확인
    p = server_config.DEFAULT_CFG
    try:
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                j = json.load(f)

            # JSON 키 매핑 (소문자 또는 대문자 키 지원)
            host = j.get("host") or j.get("MYSQL_HOST")
            database = j.get("database") or j.get("MYSQL_DATABASE")
            user = j.get("user") or j.get("MYSQL_USER")
            password = j.get("password") or j.get("MYSQL_PASSWORD")
            port = int(j.get("port") or j.get("MYSQL_PORT") or 3306)

            if host and database and user and password:
                return {"host": host, "database": database, "user": user, "password": password, "port": port}
    except Exception:
        pass

    # 3) server_config.SQL 클래스 값 사용 (Fallback)
    return {
        "host": getattr(server_config.SQL, "MYSQL_HOST", "127.0.0.1"),
        "database": getattr(server_config.SQL, "MYSQL_DATABASE", "myLucia"),
        "user": getattr(server_config.SQL, "MYSQL_USER", "root"),
        "password": getattr(server_config.SQL, "MYSQL_PASSWORD", ""),
        "port": int(getattr(server_config.SQL, "MYSQL_PORT", 3306)),
    }


class MySQLManager:
    """MySQL 데이터베이스 관리 클래스"""

    def __init__(self):
        self.connection = None
        self.cfg = _load_mysql_config()

    def connect(self) -> bool:
        """
  MySQL 데이터베이스에 연결합니다.

  Returns:
      bool: 연결 성공 여부
  """
        try:
            # 환경변수가 있으면 우선 사용, 없으면 self.cfg 사용
            self.connection = mysql.connector.connect(
                host=os.getenv("MYSQL_HOST", self.cfg["host"]),
                database=os.getenv("MYSQL_DATABASE", self.cfg["database"]),
                user=os.getenv("MYSQL_USER", self.cfg["user"]),
                password=os.getenv("MYSQL_PASSWORD", self.cfg["password"]),
                port=int(os.getenv("MYSQL_PORT", self.cfg["port"])),
            )
            if self.connection.is_connected():
                return True
            else:
                print("❌ MySQL 연결 실패 (is_connected=False)")
        except Error as e:
            print(f"❌ MySQL 연결 실패: {e}")

        self.connection = None
        return False

    def close(self):
        """MySQL 연결을 종료합니다."""
        if self.connection and self.connection.is_connected():
            self.connection.close()
        self.connection = None

    def _cursor(self, dictionary=True):
        """
  안전하게 커서를 생성하여 반환합니다.
  연결이 끊겨있으면 재연결을 시도합니다.
  """
        if not self.connection or not self.connection.is_connected():
            if not self.connect():
                return None
        try:
            return self.connection.cursor(dictionary=dictionary)
        except Exception as e:
            print(f"❌ 커서 생성 실패: {e}")
            return None

    # ─────────────────────────────────────────────────────────────
    # 비즈니스 로직 메서드
    # ─────────────────────────────────────────────────────────────

    def fetch_recent_logs(self):
        """
  최근 대화 기록을 조회합니다.
  server_config의 COMMU_LOG_INTERVAL(개수)과 COMMU_LOG_TIME(시간 표시 여부)을 따릅니다.
  """
        cur = self._cursor(dictionary=True)
        if cur is None:
            return ""

        try:
            query = """ \
                    SELECT user, userTime, assistant, assistantTime \
                    FROM logs \
                    ORDER BY userTime DESC \
                        LIMIT %s \
           """
            cur.execute(query, (server_config.LLM.COMMU_LOG_INTERVAL,))
            logs = cur.fetchall()
            cur.close()

            conversation_log = []
            conversation_log.append("다음은 멜리사와 루시아 당신의 이전 대화 기록입니다. 이를 참고하여 맥락을 유지하고 자연스러운 대화를 이어가세요\n <이전 대화 기록 시작|>")

            for log in reversed(logs):
                if server_config.LLM.COMMU_LOG_TIME:
                    conversation_log.append(f"[질문 시간 : {log['userTime']}] 멜리사 : {log['user']}")
                    conversation_log.append(f"[대답 시간 : {log['assistantTime']}] 루시아 : {log['assistant']}\n")
                else:
                    conversation_log.append(f"멜리사 : {log['user']}")
                    conversation_log.append(f"루시아 : {log['assistant']}\n")
            conversation_log.append("<이전 대화 기록 끝|>")
            self.close()
            return "\n".join(conversation_log)

        except Error as e:
            print(f"❌ 데이터 가져오기 실패: {e}")
            self.close()
            return ""

    def feedback_call(self, number):
        """특정 로그 ID에 대한 피드백 데이터를 조회합니다."""
        cur = self._cursor(dictionary=True)
        if cur is None:
            return None
        try:
            cur.execute("""
                        SELECT user, userTime, assistant, assistantTime
                        FROM logs
                        WHERE id = %s
               """, (number,))
            feedback = cur.fetchone()
            cur.close()
            self.close()
            return feedback
        except Exception as e:
            print(f"DB 에러 발생: {e}")
            self.close()
            return None

    def fetch_server_settings(self):
        """serverSettings 테이블에서 기본 설정(로그 옵션 등)을 가져옵니다."""
        cur = self._cursor(dictionary=True)
        if cur is None:
            print("❌ MySQL 연결 실패로 설정을 불러올 수 없습니다.")
            return None

        try:
            cur.execute("SELECT commu_log_time, commu_log_interval FROM serverSettings LIMIT 1;")
            result = cur.fetchone()
            cur.close()
            self.close()
            return result
        except Error as e:
            print(f"❌ 서버 설정 가져오기 실패: {e}")
            self.close()
            return None

    def fetch_server_command_settings(self):
        """serverSettings 테이블에서 캐릭터 컨셉 및 프롬프트 설정을 가져옵니다."""
        cur = self._cursor(dictionary=True)
        if cur is None:
            print("❌ MySQL 연결 실패로 설정을 불러올 수 없습니다.")
            return None

        try:
            cur.execute("SELECT character_concept, command_feedback, command_search FROM serverSettings LIMIT 1;")
            result = cur.fetchone()
            cur.close()
            self.close()
            return result
        except Error as e:
            print(f"❌ 서버 설정 가져오기 실패: {e}")
            self.close()
            return None

    def fetch_llm_settings(self):
        """serverSettings 테이블에서 LLM 관련 설정을 가져옵니다."""
        cur = self._cursor(dictionary=True)
        if cur is None:
            print("❌ MySQL 연결 실패로 설정을 불러올 수 없습니다.")
            return None

        try:
            query = """ \
                    SELECT commu_log_time, \
                           commu_log_interval, \
                           context_length, \
                           cpu_threads, \
                           llm_model_type, \
                           llm_cache_quant, \
                           llm_tensor_parallel, \
                           llm_gpu_index, \
                           llm_gpu_split, \
                           llm_temperature, \
                           llm_top_k, \
                           llm_top_p, \
                           llm_min_p, \
                           llm_repetition_penalty, \
                           llm_presence_penalty, \
                           llm_frequency_penalty \
                    FROM serverSettings LIMIT 1; \
           """
            cur.execute(query)
            result = cur.fetchone()
            cur.close()
            self.close()
            return result
        except Error as e:
            print(f"❌ LLM 설정 가져오기 실패: {e}")
            self.close()
            return None

    def fetch_tts_settings(self):
        """serverSettings 테이블에서 TTS 관련 설정을 가져옵니다."""
        cur = self._cursor(dictionary=True)
        if cur is None:
            print("❌ MySQL 연결 실패로 설정을 불러올 수 없습니다.")
            return None

        try:
            query = """ \
                    SELECT gpu_tts, \
                           tts_enable, \
                           tts_text_split_method, \
                           tts_batch_size, \
                           tts_parallel_infer, \
                           tts_split_bucket, \
                           tts_seed, \
                           tts_top_k, \
                           tts_top_p, \
                           tts_temperature, \
                           tts_repetition_penalty, \
                           tts_speed_factor, \
                           tts_language \
                    FROM serverSettings LIMIT 1; \
           """
            cur.execute(query)
            result = cur.fetchone()
            cur.close()
            self.close()
            return result
        except Error as e:
            print(f"❌ TTS 설정 가져오기 실패: {e}")
            self.close()
            return None

    def insert_default_server_settings(self):
        """serverSettings 테이블이 비어있으면 기본값을 삽입합니다."""
        cur = self._cursor(dictionary=False)
        if cur is None:
            print("❌ DB 연결 실패로 기본 설정 삽입 불가")
            return
        try:
            cur.execute("SELECT COUNT(*) FROM serverSettings WHERE id = 1;")
            count = cur.fetchone()[0] if hasattr(cur, "fetchone") else 0

            if count == 0:
                sql = """ \
                      INSERT INTO serverSettings \
                      (id, commu_log_time, commu_log_interval, character_concept, command_feedback, command_search, \
                       context_length, gpu_tts, tts_enable, cpu_threads, \
                       llm_model_type, llm_cache_quant, llm_tensor_parallel, llm_gpu_index, llm_gpu_split, \
                       llm_temperature, llm_top_k, llm_top_p, llm_min_p, llm_repetition_penalty, llm_presence_penalty, \
                       llm_frequency_penalty, \
                       tts_text_split_method, tts_batch_size, tts_parallel_infer, tts_split_bucket, tts_seed, \
                       tts_top_k, tts_top_p, tts_temperature, tts_repetition_penalty, tts_speed_factor, tts_language) \
                      VALUES (1, %s, %s, %s, %s, %s, %s, %s, %s, %s, \
                              %s, %s, %s, %s, %s, \
                              %s, %s, %s, %s, %s, %s, %s, \
                              %s, %s, %s, %s, %s, \
                              %s, %s, %s, %s, %s, %s); \
          """
                values = (
                    1 if server_config.LLM.COMMU_LOG_TIME else 0,
                    server_config.LLM.COMMU_LOG_INTERVAL,
                    "", "", "",
                    server_config.LLM.CONTEXT,
                    server_config.TTS.GPU_TTS,
                    1 if server_config.TTS.TTS_ENABLE else 0,
                    server_config.LLM.CPU_THREADS,
                    server_config.LLM.MODEL_TYPE,
                    server_config.LLM.CACHE_QUANT,
                    1 if server_config.LLM.TENSOR_PARALLEL else 0,
                    server_config.LLM.GPU_INDEX,
                    server_config.LLM.GPU_SPLIT,
                    server_config.LLM.TEMPERATURE,
                    server_config.LLM.TOP_K,
                    server_config.LLM.TOP_P,
                    server_config.LLM.MIN_P,
                    server_config.LLM.REPETITION_PENALTY,
                    server_config.LLM.PRESENCE_PENALTY,
                    server_config.LLM.FREQUENCY_PENALTY,
                    server_config.TTS.TEXT_SPLIT_METHOD,
                    server_config.TTS.BATCH_SIZE,
                    1 if server_config.TTS.PARALLEL_INFER else 0,
                    1 if server_config.TTS.SPLIT_BUCKET else 0,
                    server_config.TTS.SEED,
                    server_config.TTS.TOP_K,
                    server_config.TTS.TOP_P,
                    server_config.TTS.TEMPERATURE,
                    server_config.TTS.REPETITION_PENALTY,
                    server_config.TTS.SPEED_FACTOR,
                    server_config.TTS.TTS_LANGUAGE
                )
                cur.execute(sql, values)

            self.connection.commit()
            print("✅ 기본 설정값 삽입/유지 완료!")
        except Error as e:
            if e.errno == 1054:  # Unknown column
                print(f"⚠️ 컬럼 누락 감지 ({e}). 스키마 보정을 시도합니다.")
                self._fix_schema()
                try:
                    cur.execute(sql, values)
                    self.connection.commit()
                    print("✅ 기본 설정값 삽입 완료 (재시도 성공)")
                    return
                except Exception:
                    pass
            print(f"❌ 기본 설정 삽입 실패: {e}")
            try:
                self.connection.rollback()
            except:
                pass
        except Exception as e:
            print(f"❌ 기본 설정 삽입 실패: {e}")
            try:
                self.connection.rollback()
            except Exception:
                pass
        finally:
            self.close()

    def _fix_schema(self):
        """
  DB 스키마를 보정합니다. (컬럼 추가, 타입 변경 등)
  """
        cur = self._cursor(dictionary=False)
        if cur is None: return
        try:
            print("🔧 DB 스키마 보정 시도...")

            # 1. GPU 컬럼 타입 변경 (INT -> VARCHAR)
            try:
                cur.execute("ALTER TABLE serverSettings MODIFY COLUMN llm_gpu_index VARCHAR(255) DEFAULT '0';")
                cur.execute("ALTER TABLE serverSettings MODIFY COLUMN llm_gpu_split VARCHAR(255) DEFAULT '';")
            except Exception:
                pass

            # 2. TTS Enable 컬럼 추가
            try:
                cur.execute("ALTER TABLE serverSettings ADD COLUMN tts_enable TINYINT(1) DEFAULT 0;")
            except Exception:
                pass

            self.connection.commit()
            print("✅ 스키마 보정 완료")
        except Exception as e:
            print(f"❌ 스키마 보정 실패: {e}")
        finally:
            cur.close()

    def save_command(self):
        """
  현재 server_config의 모든 설정을 serverSettings 테이블에 저장(UPSERT)합니다.
  """
        cur = self._cursor(dictionary=False)
        if cur is None:
            print("❌ MySQL 연결 실패로 설정 저장 불가")
            return None

        sql = """ \
              INSERT INTO serverSettings \
              (id, commu_log_time, commu_log_interval, character_concept, command_feedback, command_search, \
               context_length, gpu_tts, tts_enable, cpu_threads, \
               llm_model_type, llm_cache_quant, llm_tensor_parallel, llm_gpu_index, llm_gpu_split, \
               llm_temperature, llm_top_k, llm_top_p, llm_min_p, llm_repetition_penalty, llm_presence_penalty, \
               llm_frequency_penalty, \
               tts_text_split_method, tts_batch_size, tts_parallel_infer, tts_split_bucket, tts_seed, \
               tts_top_k, tts_top_p, tts_temperature, tts_repetition_penalty, tts_speed_factor, tts_language) \
              VALUES (1, %s, %s, %s, %s, %s, %s, %s, %s, %s, \
                      %s, %s, %s, %s, %s, \
                      %s, %s, %s, %s, %s, %s, %s, \
                      %s, %s, %s, %s, %s, \
                      %s, %s, %s, %s, %s, %s) ON DUPLICATE KEY \
              UPDATE \
                  commu_log_time = \
              VALUES (commu_log_time), commu_log_interval = \
              VALUES (commu_log_interval), character_concept = \
              VALUES (character_concept), command_feedback = \
              VALUES (command_feedback), command_search = \
              VALUES (command_search), context_length = \
              VALUES (context_length), gpu_tts = \
              VALUES (gpu_tts), tts_enable = \
              VALUES (tts_enable), cpu_threads = \
              VALUES (cpu_threads), llm_model_type = \
              VALUES (llm_model_type), llm_cache_quant = \
              VALUES (llm_cache_quant), llm_tensor_parallel = \
              VALUES (llm_tensor_parallel), llm_gpu_index = \
              VALUES (llm_gpu_index), llm_gpu_split = \
              VALUES (llm_gpu_split), llm_temperature = \
              VALUES (llm_temperature), llm_top_k = \
              VALUES (llm_top_k), llm_top_p = \
              VALUES (llm_top_p), llm_min_p = \
              VALUES (llm_min_p), llm_repetition_penalty = \
              VALUES (llm_repetition_penalty), llm_presence_penalty = \
              VALUES (llm_presence_penalty), llm_frequency_penalty = \
              VALUES (llm_frequency_penalty), tts_text_split_method = \
              VALUES (tts_text_split_method), tts_batch_size = \
              VALUES (tts_batch_size), tts_parallel_infer = \
              VALUES (tts_parallel_infer), tts_split_bucket = \
              VALUES (tts_split_bucket), tts_seed = \
              VALUES (tts_seed), tts_top_k = \
              VALUES (tts_top_k), tts_top_p = \
              VALUES (tts_top_p), tts_temperature = \
              VALUES (tts_temperature), tts_repetition_penalty = \
              VALUES (tts_repetition_penalty), tts_speed_factor = \
              VALUES (tts_speed_factor), tts_language = \
              VALUES (tts_language); \
        """
        values = (
            1 if server_config.LLM.COMMU_LOG_TIME else 0,
            server_config.LLM.COMMU_LOG_INTERVAL,
            server_config.LLM.LLM_CHAT_FORMAT,
            server_config.LLM.LLM_FEEDBACK_FORMAT,
            server_config.LLM.LLM_RAG_SEARCH_FORMAT,
            server_config.LLM.CONTEXT,
            server_config.TTS.GPU_TTS,
            1 if server_config.TTS.TTS_ENABLE else 0,
            server_config.LLM.CPU_THREADS,
            server_config.LLM.MODEL_TYPE,
            server_config.LLM.CACHE_QUANT,
            1 if server_config.LLM.TENSOR_PARALLEL else 0,
            server_config.LLM.GPU_INDEX,
            server_config.LLM.GPU_SPLIT,
            server_config.LLM.TEMPERATURE,
            server_config.LLM.TOP_K,
            server_config.LLM.TOP_P,
            server_config.LLM.MIN_P,
            server_config.LLM.REPETITION_PENALTY,
            server_config.LLM.PRESENCE_PENALTY,
            server_config.LLM.FREQUENCY_PENALTY,
            server_config.TTS.TEXT_SPLIT_METHOD,
            server_config.TTS.BATCH_SIZE,
            1 if server_config.TTS.PARALLEL_INFER else 0,
            1 if server_config.TTS.SPLIT_BUCKET else 0,
            server_config.TTS.SEED,
            server_config.TTS.TOP_K,
            server_config.TTS.TOP_P,
            server_config.TTS.TEMPERATURE,
            server_config.TTS.REPETITION_PENALTY,
            server_config.TTS.SPEED_FACTOR,
            server_config.TTS.TTS_LANGUAGE
        )

        try:
            cur.execute(sql, values)
            self.connection.commit()
            print("✅ serverSettings 갱신/생성 완료")
            return True

        except Error as e:
            # 1265: Data truncated, 1366: Incorrect integer value, 1054: Unknown column
            if e.errno in (1265, 1366, 1054):
                print(f"⚠️ DB 스키마 문제 감지 ({e}). 보정을 시도합니다.")
                self._fix_schema()
                # 재시도
                try:
                    cur.close()
                    cur = self._cursor(dictionary=False)
                    cur.execute(sql, values)
                    self.connection.commit()
                    print("✅ serverSettings 갱신/생성 완료 (재시도 성공)")
                    return True
                except Exception as retry_e:
                    print(f"❌ 재시도 실패: {retry_e}")

            try:
                self.connection.rollback()
            except Exception:
                pass
            print(f"❌ 서버 설정 저장 실패: {e}")
            return None

        finally:
            try:
                cur.close()
            except Exception:
                pass
            self.close()

    def test_mysql_connection(self, ui):
        """UI에서 입력한 정보로 MySQL 연결 테스트를 수행합니다."""
        # 1) UI에서 값 읽기
        host = ui.IPLineEdit.text().strip()
        database = ui.DBLineEdit.text().strip()
        user = ui.IDLineEdit.text().strip()
        password = ui.PWLineEdit.text().strip()
        port_text = ui.PortLineEdit.text().strip()

        try:
            port = int(port_text)
        except ValueError:
            ui.TestResultLabel.setText("최종 결과 : FAIL (포트 오류)")
            return

        ui.TestIPLabel.setText(f"IP : {host}")
        ui.TestDbLabel.setText(f"DB : {database}")
        ui.TestIDLabel.setText(f"ID/PW : {user} / {password}")
        ui.TestPortLabel.setText(f"PORT : {port_text}")

        # 2) 직접 접속 테스트
        try:
            conn = mysql.connector.connect(
                host=host,
                database=database,
                user=user,
                password=password,
                port=port,
            )
            if conn.is_connected():
                print("🔍 테스트: MySQL 연결 OK")
                conn.close()
                ui.TestResultLabel.setText("최종 결과 : OK")
            else:
                print("🔍 테스트: MySQL 연결 실패 (연결 객체 없음)")
                ui.TestResultLabel.setText("최종 결과 : FAIL")
        except Error as e:
            print(f"🔍 테스트: MySQL 연결 실패 → {e}")
            ui.TestResultLabel.setText("최종 결과 : FAIL")
