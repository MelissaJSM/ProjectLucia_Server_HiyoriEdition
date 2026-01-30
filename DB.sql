-- MySQL dump 10.13  Distrib 8.0.41, for Win64 (x86_64)
--
-- Host: 192.168.35.97    Database: myLucia
-- ------------------------------------------------------
-- Server version	5.5.5-10.11.14-MariaDB-0+deb12u2

/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!50503 SET NAMES utf8 */;
/*!40103 SET @OLD_TIME_ZONE=@@TIME_ZONE */;
/*!40103 SET TIME_ZONE='+00:00' */;
/*!40014 SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;

--
-- Table structure for table `logs`
--

DROP TABLE IF EXISTS `logs`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `logs` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `user` text NOT NULL,
  `assistant` text NOT NULL,
  `emotion` varchar(50) DEFAULT NULL,
  `userTime` timestamp NULL DEFAULT current_timestamp(),
  `assistantTime` timestamp NULL DEFAULT current_timestamp(),
  `isFeedback` tinyint(4) NOT NULL DEFAULT 0,
  `isLearning` tinyint(4) NOT NULL DEFAULT 0,
  `feedbackData` text DEFAULT NULL,
  `feedbackHint` text DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=264 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `logs`
--

LOCK TABLES `logs` WRITE;
/*!40000 ALTER TABLE `logs` DISABLE KEYS */;
/*!40000 ALTER TABLE `logs` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `serverSettings`
--

DROP TABLE IF EXISTS `serverSettings`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `serverSettings` (
  `id` int(11) NOT NULL,
  `commu_log_interval` int(11) DEFAULT 10,
  `commu_log_time` tinyint(1) DEFAULT 0,
  `character_concept` text DEFAULT NULL,
  `command_feedback` text DEFAULT NULL,
  `command_search` text DEFAULT NULL,
  `context_length` int(11) DEFAULT 2048,
  `gpu_tts` int(11) DEFAULT 0,
  `cpu_threads` int(11) DEFAULT 4,
  `llm_model_type` varchar(50) DEFAULT NULL,
  `llm_cache_quant` varchar(20) DEFAULT NULL,
  `llm_tensor_parallel` tinyint(1) DEFAULT 0,
  `llm_gpu_index` varchar(255) DEFAULT '0',
  `llm_gpu_split` varchar(255) DEFAULT '',
  `llm_temperature` float DEFAULT 1,
  `llm_top_k` int(11) DEFAULT 50,
  `llm_top_p` float DEFAULT 1,
  `llm_min_p` float DEFAULT 0,
  `llm_repetition_penalty` float DEFAULT 1,
  `llm_presence_penalty` float DEFAULT 0,
  `llm_frequency_penalty` float DEFAULT 0,
  `tts_enable` tinyint(1) DEFAULT 1,
  `tts_text_split_method` varchar(20) DEFAULT NULL,
  `tts_batch_size` int(11) DEFAULT 1,
  `tts_parallel_infer` tinyint(1) DEFAULT 0,
  `tts_split_bucket` tinyint(1) DEFAULT 0,
  `tts_seed` bigint(20) DEFAULT -1,
  `tts_top_k` int(11) DEFAULT 20,
  `tts_top_p` float DEFAULT 0.8,
  `tts_temperature` float DEFAULT 0.8,
  `tts_repetition_penalty` float DEFAULT 1.35,
  `tts_speed_factor` float DEFAULT 1,
  `tts_language` varchar(10) DEFAULT 'ko',
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `serverSettings`
--

LOCK TABLES `serverSettings` WRITE;
/*!40000 ALTER TABLE `serverSettings` DISABLE KEYS */;
/*!40000 ALTER TABLE `serverSettings` ENABLE KEYS */;
UNLOCK TABLES;
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2026-01-29 11:27:59
