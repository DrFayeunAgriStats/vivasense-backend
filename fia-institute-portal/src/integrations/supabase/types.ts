export type Json =
  | string
  | number
  | boolean
  | null
  | { [key: string]: Json | undefined }
  | Json[]

export type Database = {
  // Allows to automatically instantiate createClient with right options
  // instead of createClient<Database, { PostgrestVersion: 'XX' }>(URL, KEY)
  __InternalSupabase: {
    PostgrestVersion: "14.1"
  }
  public: {
    Tables: {
      adap_student_codes: {
        Row: {
          code: string
          created_at: string
          id: string
          is_admin: boolean
          status: string
        }
        Insert: {
          code: string
          created_at?: string
          id?: string
          is_admin?: boolean
          status?: string
        }
        Update: {
          code?: string
          created_at?: string
          id?: string
          is_admin?: boolean
          status?: string
        }
        Relationships: []
      }
      adap_students: {
        Row: {
          certificate_code: string | null
          chat_history: Json
          cohort: string
          completed_weeks: number[]
          created_at: string
          current_week: number
          full_name: string
          id: string
          last_active: string
          last_quiz_score: string | null
          student_id: string
          updated_at: string
        }
        Insert: {
          certificate_code?: string | null
          chat_history?: Json
          cohort: string
          completed_weeks?: number[]
          created_at?: string
          current_week?: number
          full_name: string
          id?: string
          last_active?: string
          last_quiz_score?: string | null
          student_id: string
          updated_at?: string
        }
        Update: {
          certificate_code?: string | null
          chat_history?: Json
          cohort?: string
          completed_weeks?: number[]
          created_at?: string
          current_week?: number
          full_name?: string
          id?: string
          last_active?: string
          last_quiz_score?: string | null
          student_id?: string
          updated_at?: string
        }
        Relationships: []
      }
      ai_conversations: {
        Row: {
          context: Json | null
          created_at: string
          id: string
          mode: string
          title: string | null
          updated_at: string
          user_id: string
        }
        Insert: {
          context?: Json | null
          created_at?: string
          id?: string
          mode?: string
          title?: string | null
          updated_at?: string
          user_id: string
        }
        Update: {
          context?: Json | null
          created_at?: string
          id?: string
          mode?: string
          title?: string | null
          updated_at?: string
          user_id?: string
        }
        Relationships: []
      }
      ai_messages: {
        Row: {
          content: string
          conversation_id: string
          created_at: string
          id: string
          mode: string | null
          role: string
        }
        Insert: {
          content: string
          conversation_id: string
          created_at?: string
          id?: string
          mode?: string | null
          role: string
        }
        Update: {
          content?: string
          conversation_id?: string
          created_at?: string
          id?: string
          mode?: string | null
          role?: string
        }
        Relationships: [
          {
            foreignKeyName: "ai_messages_conversation_id_fkey"
            columns: ["conversation_id"]
            isOneToOne: false
            referencedRelation: "ai_conversations"
            referencedColumns: ["id"]
          },
        ]
      }
      ai_usage_logs: {
        Row: {
          created_at: string
          id: string
          mode: string
          token_count: number | null
          user_id: string
        }
        Insert: {
          created_at?: string
          id?: string
          mode: string
          token_count?: number | null
          user_id: string
        }
        Update: {
          created_at?: string
          id?: string
          mode?: string
          token_count?: number | null
          user_id?: string
        }
        Relationships: []
      }
      analysis_uploads: {
        Row: {
          analysis_type: string
          created_at: string
          id: string
          input_data: string
          input_method: string
          user_id: string
        }
        Insert: {
          analysis_type: string
          created_at?: string
          id?: string
          input_data: string
          input_method: string
          user_id: string
        }
        Update: {
          analysis_type?: string
          created_at?: string
          id?: string
          input_data?: string
          input_method?: string
          user_id?: string
        }
        Relationships: []
      }
      audit_logs: {
        Row: {
          action_description: string
          action_type: string
          created_at: string
          id: string
          metadata: Json | null
          target_user_id: string | null
          user_id: string
        }
        Insert: {
          action_description: string
          action_type: string
          created_at?: string
          id?: string
          metadata?: Json | null
          target_user_id?: string | null
          user_id: string
        }
        Update: {
          action_description?: string
          action_type?: string
          created_at?: string
          id?: string
          metadata?: Json | null
          target_user_id?: string | null
          user_id?: string
        }
        Relationships: []
      }
      badges: {
        Row: {
          created_at: string
          description: string | null
          icon: string | null
          id: string
          key: string
          label: string
        }
        Insert: {
          created_at?: string
          description?: string | null
          icon?: string | null
          id?: string
          key: string
          label: string
        }
        Update: {
          created_at?: string
          description?: string | null
          icon?: string | null
          id?: string
          key?: string
          label?: string
        }
        Relationships: []
      }
      bgm_students: {
        Row: {
          best_scores: Json
          code: string
          completed_modules: number[]
          completion_token: string | null
          created_at: string
          current_module: number
          full_name: string
          id: string
          progress_percent: number
          r_practice_status: Json
          registration_id: string
          token_status: string
          updated_at: string
        }
        Insert: {
          best_scores?: Json
          code: string
          completed_modules?: number[]
          completion_token?: string | null
          created_at?: string
          current_module?: number
          full_name: string
          id?: string
          progress_percent?: number
          r_practice_status?: Json
          registration_id: string
          token_status?: string
          updated_at?: string
        }
        Update: {
          best_scores?: Json
          code?: string
          completed_modules?: number[]
          completion_token?: string | null
          created_at?: string
          current_module?: number
          full_name?: string
          id?: string
          progress_percent?: number
          r_practice_status?: Json
          registration_id?: string
          token_status?: string
          updated_at?: string
        }
        Relationships: [
          {
            foreignKeyName: "bgm_students_code_fkey"
            columns: ["code"]
            isOneToOne: true
            referencedRelation: "student_codes"
            referencedColumns: ["code"]
          },
        ]
      }
      booking_requests: {
        Row: {
          approved_datetime: string | null
          booking_status: string
          created_at: string
          draft_reference: string | null
          id: string
          milestone_type: string
          requested_date: string
          student_id: string
          student_note: string | null
          supervisor_id: string
          supervisor_note: string | null
          updated_at: string
        }
        Insert: {
          approved_datetime?: string | null
          booking_status?: string
          created_at?: string
          draft_reference?: string | null
          id?: string
          milestone_type: string
          requested_date: string
          student_id: string
          student_note?: string | null
          supervisor_id: string
          supervisor_note?: string | null
          updated_at?: string
        }
        Update: {
          approved_datetime?: string | null
          booking_status?: string
          created_at?: string
          draft_reference?: string | null
          id?: string
          milestone_type?: string
          requested_date?: string
          student_id?: string
          student_note?: string | null
          supervisor_id?: string
          supervisor_note?: string | null
          updated_at?: string
        }
        Relationships: []
      }
      certificate_eligibility: {
        Row: {
          certificate_status: string
          created_at: string
          defense_completed: boolean
          defense_score: number | null
          id: string
          integrity_accepted: boolean
          milestones_completed: boolean
          modules_completed: boolean
          reviewed_at: string | null
          reviewed_by: string | null
          supervisor_flags_resolved: boolean
          updated_at: string
          user_id: string
        }
        Insert: {
          certificate_status?: string
          created_at?: string
          defense_completed?: boolean
          defense_score?: number | null
          id?: string
          integrity_accepted?: boolean
          milestones_completed?: boolean
          modules_completed?: boolean
          reviewed_at?: string | null
          reviewed_by?: string | null
          supervisor_flags_resolved?: boolean
          updated_at?: string
          user_id: string
        }
        Update: {
          certificate_status?: string
          created_at?: string
          defense_completed?: boolean
          defense_score?: number | null
          id?: string
          integrity_accepted?: boolean
          milestones_completed?: boolean
          modules_completed?: boolean
          reviewed_at?: string | null
          reviewed_by?: string | null
          supervisor_flags_resolved?: boolean
          updated_at?: string
          user_id?: string
        }
        Relationships: []
      }
      certificates: {
        Row: {
          appreciation_text: string | null
          certificate_id: string
          certificate_type: string
          certify_phrase: string
          competencies: string[] | null
          completion_phrase: string
          created_at: string
          id: string
          issue_date: string
          program_name: string
          recipient_name: string
          signature_url: string | null
          status: string
          training_end_date: string | null
          training_start_date: string | null
          updated_at: string
        }
        Insert: {
          appreciation_text?: string | null
          certificate_id: string
          certificate_type?: string
          certify_phrase?: string
          competencies?: string[] | null
          completion_phrase?: string
          created_at?: string
          id?: string
          issue_date?: string
          program_name?: string
          recipient_name: string
          signature_url?: string | null
          status?: string
          training_end_date?: string | null
          training_start_date?: string | null
          updated_at?: string
        }
        Update: {
          appreciation_text?: string | null
          certificate_id?: string
          certificate_type?: string
          certify_phrase?: string
          competencies?: string[] | null
          completion_phrase?: string
          created_at?: string
          id?: string
          issue_date?: string
          program_name?: string
          recipient_name?: string
          signature_url?: string | null
          status?: string
          training_end_date?: string | null
          training_start_date?: string | null
          updated_at?: string
        }
        Relationships: []
      }
      competency_scores: {
        Row: {
          category: string
          id: string
          max_score: number
          score: number
          updated_at: string
          user_id: string
        }
        Insert: {
          category: string
          id?: string
          max_score?: number
          score?: number
          updated_at?: string
          user_id: string
        }
        Update: {
          category?: string
          id?: string
          max_score?: number
          score?: number
          updated_at?: string
          user_id?: string
        }
        Relationships: []
      }
      daily_drill_attempts: {
        Row: {
          ai_feedback: string | null
          completed_at: string | null
          created_at: string
          drill_content: Json
          drill_type: string
          id: string
          score: number | null
          student_response: string | null
          user_id: string
        }
        Insert: {
          ai_feedback?: string | null
          completed_at?: string | null
          created_at?: string
          drill_content?: Json
          drill_type: string
          id?: string
          score?: number | null
          student_response?: string | null
          user_id: string
        }
        Update: {
          ai_feedback?: string | null
          completed_at?: string | null
          created_at?: string
          drill_content?: Json
          drill_type?: string
          id?: string
          score?: number | null
          student_response?: string | null
          user_id?: string
        }
        Relationships: []
      }
      defense_simulator_attempts: {
        Row: {
          ai_feedback_summary: string | null
          completed_at: string | null
          created_at: string
          id: string
          scores: Json
          simulation_type: string
          total_score: number | null
          track: string
          updated_at: string
          user_id: string
        }
        Insert: {
          ai_feedback_summary?: string | null
          completed_at?: string | null
          created_at?: string
          id?: string
          scores?: Json
          simulation_type?: string
          total_score?: number | null
          track: string
          updated_at?: string
          user_id: string
        }
        Update: {
          ai_feedback_summary?: string | null
          completed_at?: string | null
          created_at?: string
          id?: string
          scores?: Json
          simulation_type?: string
          total_score?: number | null
          track?: string
          updated_at?: string
          user_id?: string
        }
        Relationships: []
      }
      defense_simulator_messages: {
        Row: {
          attempt_id: string
          content: string
          created_at: string
          examiner_role: string | null
          id: string
          role: string
        }
        Insert: {
          attempt_id: string
          content: string
          created_at?: string
          examiner_role?: string | null
          id?: string
          role: string
        }
        Update: {
          attempt_id?: string
          content?: string
          created_at?: string
          examiner_role?: string | null
          id?: string
          role?: string
        }
        Relationships: [
          {
            foreignKeyName: "defense_simulator_messages_attempt_id_fkey"
            columns: ["attempt_id"]
            isOneToOne: false
            referencedRelation: "defense_simulator_attempts"
            referencedColumns: ["id"]
          },
        ]
      }
      diagnostic_attempts: {
        Row: {
          completed_at: string
          id: string
          level: Database["public"]["Enums"]["diagnostic_level"]
          recommended_starting_point: string | null
          score: number
          total_questions: number
          user_id: string
        }
        Insert: {
          completed_at?: string
          id?: string
          level: Database["public"]["Enums"]["diagnostic_level"]
          recommended_starting_point?: string | null
          score: number
          total_questions: number
          user_id: string
        }
        Update: {
          completed_at?: string
          id?: string
          level?: Database["public"]["Enums"]["diagnostic_level"]
          recommended_starting_point?: string | null
          score?: number
          total_questions?: number
          user_id?: string
        }
        Relationships: []
      }
      diagnostic_responses: {
        Row: {
          answer: string
          attempt_id: string
          created_at: string
          id: string
          is_correct: boolean
          question_index: number
        }
        Insert: {
          answer: string
          attempt_id: string
          created_at?: string
          id?: string
          is_correct: boolean
          question_index: number
        }
        Update: {
          answer?: string
          attempt_id?: string
          created_at?: string
          id?: string
          is_correct?: boolean
          question_index?: number
        }
        Relationships: [
          {
            foreignKeyName: "diagnostic_responses_attempt_id_fkey"
            columns: ["attempt_id"]
            isOneToOne: false
            referencedRelation: "diagnostic_attempts"
            referencedColumns: ["id"]
          },
        ]
      }
      institution_accounts: {
        Row: {
          contact_email: string
          country: string | null
          created_at: string
          id: string
          is_active: boolean
          max_students: number | null
          name: string
          subscription_tier: string | null
          updated_at: string
        }
        Insert: {
          contact_email: string
          country?: string | null
          created_at?: string
          id?: string
          is_active?: boolean
          max_students?: number | null
          name: string
          subscription_tier?: string | null
          updated_at?: string
        }
        Update: {
          contact_email?: string
          country?: string | null
          created_at?: string
          id?: string
          is_active?: boolean
          max_students?: number | null
          name?: string
          subscription_tier?: string | null
          updated_at?: string
        }
        Relationships: []
      }
      institution_students: {
        Row: {
          enrolled_at: string
          id: string
          institution_id: string
          user_id: string
        }
        Insert: {
          enrolled_at?: string
          id?: string
          institution_id: string
          user_id: string
        }
        Update: {
          enrolled_at?: string
          id?: string
          institution_id?: string
          user_id?: string
        }
        Relationships: [
          {
            foreignKeyName: "institution_students_institution_id_fkey"
            columns: ["institution_id"]
            isOneToOne: false
            referencedRelation: "institution_accounts"
            referencedColumns: ["id"]
          },
        ]
      }
      interpretation_responses: {
        Row: {
          ai_feedback: string | null
          created_at: string
          id: string
          level: number
          question: string
          student_response: string | null
          updated_at: string
          upload_id: string
          user_id: string
        }
        Insert: {
          ai_feedback?: string | null
          created_at?: string
          id?: string
          level: number
          question: string
          student_response?: string | null
          updated_at?: string
          upload_id: string
          user_id: string
        }
        Update: {
          ai_feedback?: string | null
          created_at?: string
          id?: string
          level?: number
          question?: string
          student_response?: string | null
          updated_at?: string
          upload_id?: string
          user_id?: string
        }
        Relationships: [
          {
            foreignKeyName: "interpretation_responses_upload_id_fkey"
            columns: ["upload_id"]
            isOneToOne: false
            referencedRelation: "analysis_uploads"
            referencedColumns: ["id"]
          },
        ]
      }
      journal_articles: {
        Row: {
          abstract: string | null
          authors: string
          created_at: string
          doi_url: string | null
          id: string
          is_current_issue: boolean | null
          issue: number | null
          published_at: string | null
          title: string
          volume: number | null
        }
        Insert: {
          abstract?: string | null
          authors: string
          created_at?: string
          doi_url?: string | null
          id?: string
          is_current_issue?: boolean | null
          issue?: number | null
          published_at?: string | null
          title: string
          volume?: number | null
        }
        Update: {
          abstract?: string | null
          authors?: string
          created_at?: string
          doi_url?: string | null
          id?: string
          is_current_issue?: boolean | null
          issue?: number | null
          published_at?: string | null
          title?: string
          volume?: number | null
        }
        Relationships: []
      }
      journal_submissions: {
        Row: {
          abstract: string
          author_name: string
          created_at: string
          email: string
          file_path: string | null
          id: string
          manuscript_title: string
          status: string | null
        }
        Insert: {
          abstract: string
          author_name: string
          created_at?: string
          email: string
          file_path?: string | null
          id?: string
          manuscript_title: string
          status?: string | null
        }
        Update: {
          abstract?: string
          author_name?: string
          created_at?: string
          email?: string
          file_path?: string | null
          id?: string
          manuscript_title?: string
          status?: string | null
        }
        Relationships: []
      }
      milestones: {
        Row: {
          completed_at: string | null
          created_at: string
          description: string | null
          id: string
          is_completed: boolean
          stage: string
          title: string
          user_id: string
        }
        Insert: {
          completed_at?: string | null
          created_at?: string
          description?: string | null
          id?: string
          is_completed?: boolean
          stage: string
          title: string
          user_id: string
        }
        Update: {
          completed_at?: string | null
          created_at?: string
          description?: string | null
          id?: string
          is_completed?: boolean
          stage?: string
          title?: string
          user_id?: string
        }
        Relationships: []
      }
      notifications: {
        Row: {
          created_at: string
          id: string
          is_read: boolean
          link: string | null
          message: string
          title: string
          type: string
          user_id: string
        }
        Insert: {
          created_at?: string
          id?: string
          is_read?: boolean
          link?: string | null
          message: string
          title: string
          type?: string
          user_id: string
        }
        Update: {
          created_at?: string
          id?: string
          is_read?: boolean
          link?: string | null
          message?: string
          title?: string
          type?: string
          user_id?: string
        }
        Relationships: []
      }
      page_visits: {
        Row: {
          id: string
          page_path: string
          referrer: string | null
          user_agent: string | null
          visited_at: string
        }
        Insert: {
          id?: string
          page_path: string
          referrer?: string | null
          user_agent?: string | null
          visited_at?: string
        }
        Update: {
          id?: string
          page_path?: string
          referrer?: string | null
          user_agent?: string | null
          visited_at?: string
        }
        Relationships: []
      }
      profiles: {
        Row: {
          academic_track: Database["public"]["Enums"]["academic_track"] | null
          country: string | null
          created_at: string
          current_research_stage:
            | Database["public"]["Enums"]["research_stage"]
            | null
          diagnostic_level:
            | Database["public"]["Enums"]["diagnostic_level"]
            | null
          diagnostic_score: number | null
          discipline: string | null
          email: string
          full_name: string
          id: string
          institution: string | null
          onboarding_completed: boolean
          plan: string | null
          plan_activated_at: string | null
          plan_notes: string | null
          portfolio_share_token: string | null
          portfolio_visibility: string
          pro_interest: boolean | null
          pro_interest_date: string | null
          updated_at: string
        }
        Insert: {
          academic_track?: Database["public"]["Enums"]["academic_track"] | null
          country?: string | null
          created_at?: string
          current_research_stage?:
            | Database["public"]["Enums"]["research_stage"]
            | null
          diagnostic_level?:
            | Database["public"]["Enums"]["diagnostic_level"]
            | null
          diagnostic_score?: number | null
          discipline?: string | null
          email: string
          full_name: string
          id: string
          institution?: string | null
          onboarding_completed?: boolean
          plan?: string | null
          plan_activated_at?: string | null
          plan_notes?: string | null
          portfolio_share_token?: string | null
          portfolio_visibility?: string
          pro_interest?: boolean | null
          pro_interest_date?: string | null
          updated_at?: string
        }
        Update: {
          academic_track?: Database["public"]["Enums"]["academic_track"] | null
          country?: string | null
          created_at?: string
          current_research_stage?:
            | Database["public"]["Enums"]["research_stage"]
            | null
          diagnostic_level?:
            | Database["public"]["Enums"]["diagnostic_level"]
            | null
          diagnostic_score?: number | null
          discipline?: string | null
          email?: string
          full_name?: string
          id?: string
          institution?: string | null
          onboarding_completed?: boolean
          plan?: string | null
          plan_activated_at?: string | null
          plan_notes?: string | null
          portfolio_share_token?: string | null
          portfolio_visibility?: string
          pro_interest?: boolean | null
          pro_interest_date?: string | null
          updated_at?: string
        }
        Relationships: []
      }
      analysis_logs: {
        Row: {
          analysis_type: string
          created_at: string
          dataset_rows: number | null
          design_type: string | null
          duration_ms: number | null
          error_message: string | null
          id: string
          success: boolean
          trait_count: number | null
          user_id: string
        }
        Insert: {
          analysis_type: string
          created_at?: string
          dataset_rows?: number | null
          design_type?: string | null
          duration_ms?: number | null
          error_message?: string | null
          id?: string
          success?: boolean
          trait_count?: number | null
          user_id: string
        }
        Update: {
          analysis_type?: string
          created_at?: string
          dataset_rows?: number | null
          design_type?: string | null
          duration_ms?: number | null
          error_message?: string | null
          id?: string
          success?: boolean
          trait_count?: number | null
          user_id?: string
        }
        Relationships: []
      }
      progress_records: {
        Row: {
          completed_at: string | null
          id: string
          module_key: string
          progress_percent: number
          started_at: string | null
          status: string
          updated_at: string
          user_id: string
        }
        Insert: {
          completed_at?: string | null
          id?: string
          module_key: string
          progress_percent?: number
          started_at?: string | null
          status?: string
          updated_at?: string
          user_id: string
        }
        Update: {
          completed_at?: string | null
          id?: string
          module_key?: string
          progress_percent?: number
          started_at?: string | null
          status?: string
          updated_at?: string
          user_id?: string
        }
        Relationships: []
      }
      student_codes: {
        Row: {
          code: string
          created_at: string
          id: string
          is_admin: boolean
          status: string
        }
        Insert: {
          code: string
          created_at?: string
          id?: string
          is_admin?: boolean
          status?: string
        }
        Update: {
          code?: string
          created_at?: string
          id?: string
          is_admin?: boolean
          status?: string
        }
        Relationships: []
      }
      subscriptions: {
        Row: {
          created_at: string
          id: string
          payment_date: string | null
          payment_reference: string | null
          payment_status: string | null
          renewal_date: string | null
          subscription_status: string
          tier: string
          updated_at: string
          user_id: string
        }
        Insert: {
          created_at?: string
          id?: string
          payment_date?: string | null
          payment_reference?: string | null
          payment_status?: string | null
          renewal_date?: string | null
          subscription_status?: string
          tier?: string
          updated_at?: string
          user_id: string
        }
        Update: {
          created_at?: string
          id?: string
          payment_date?: string | null
          payment_reference?: string | null
          payment_status?: string | null
          renewal_date?: string | null
          subscription_status?: string
          tier?: string
          updated_at?: string
          user_id?: string
        }
        Relationships: []
      }
      supervisor_assignments: {
        Row: {
          assigned_at: string
          assignment_status: string
          created_at: string
          id: string
          student_id: string
          supervisor_id: string
          updated_at: string
        }
        Insert: {
          assigned_at?: string
          assignment_status?: string
          created_at?: string
          id?: string
          student_id: string
          supervisor_id: string
          updated_at?: string
        }
        Update: {
          assigned_at?: string
          assignment_status?: string
          created_at?: string
          id?: string
          student_id?: string
          supervisor_id?: string
          updated_at?: string
        }
        Relationships: []
      }
      supervisor_flags: {
        Row: {
          created_at: string
          description: string | null
          flag_type: string
          id: string
          is_resolved: boolean
          resolved_at: string | null
          student_id: string
          supervisor_id: string
        }
        Insert: {
          created_at?: string
          description?: string | null
          flag_type: string
          id?: string
          is_resolved?: boolean
          resolved_at?: string | null
          student_id: string
          supervisor_id: string
        }
        Update: {
          created_at?: string
          description?: string | null
          flag_type?: string
          id?: string
          is_resolved?: boolean
          resolved_at?: string | null
          student_id?: string
          supervisor_id?: string
        }
        Relationships: []
      }
      supervisor_overrides: {
        Row: {
          created_at: string
          id: string
          milestone_id: string | null
          new_status: string | null
          original_status: string | null
          override_reason: string | null
          override_type: string
          student_id: string
          supervisor_comment: string | null
          supervisor_id: string
        }
        Insert: {
          created_at?: string
          id?: string
          milestone_id?: string | null
          new_status?: string | null
          original_status?: string | null
          override_reason?: string | null
          override_type: string
          student_id: string
          supervisor_comment?: string | null
          supervisor_id: string
        }
        Update: {
          created_at?: string
          id?: string
          milestone_id?: string | null
          new_status?: string | null
          original_status?: string | null
          override_reason?: string | null
          override_type?: string
          student_id?: string
          supervisor_comment?: string | null
          supervisor_id?: string
        }
        Relationships: [
          {
            foreignKeyName: "supervisor_overrides_milestone_id_fkey"
            columns: ["milestone_id"]
            isOneToOne: false
            referencedRelation: "milestones"
            referencedColumns: ["id"]
          },
        ]
      }
      user_badges: {
        Row: {
          badge_id: string
          earned_at: string
          id: string
          user_id: string
        }
        Insert: {
          badge_id: string
          earned_at?: string
          id?: string
          user_id: string
        }
        Update: {
          badge_id?: string
          earned_at?: string
          id?: string
          user_id?: string
        }
        Relationships: [
          {
            foreignKeyName: "user_badges_badge_id_fkey"
            columns: ["badge_id"]
            isOneToOne: false
            referencedRelation: "badges"
            referencedColumns: ["id"]
          },
        ]
      }
      user_roles: {
        Row: {
          created_at: string
          id: string
          role: Database["public"]["Enums"]["app_role"]
          user_id: string
        }
        Insert: {
          created_at?: string
          id?: string
          role: Database["public"]["Enums"]["app_role"]
          user_id: string
        }
        Update: {
          created_at?: string
          id?: string
          role?: Database["public"]["Enums"]["app_role"]
          user_id?: string
        }
        Relationships: []
      }
      user_streaks: {
        Row: {
          current_streak: number
          id: string
          last_activity_date: string | null
          longest_streak: number
          updated_at: string
          user_id: string
        }
        Insert: {
          current_streak?: number
          id?: string
          last_activity_date?: string | null
          longest_streak?: number
          updated_at?: string
          user_id: string
        }
        Update: {
          current_streak?: number
          id?: string
          last_activity_date?: string | null
          longest_streak?: number
          updated_at?: string
          user_id?: string
        }
        Relationships: []
      }
    }
    Views: {
      [_ in never]: never
    }
    Functions: {
      has_role: {
        Args: {
          _role: Database["public"]["Enums"]["app_role"]
          _user_id: string
        }
        Returns: boolean
      }
    }
    Enums: {
      academic_track:
        | "undergraduate_project"
        | "msc_thesis"
        | "phd_research"
        | "research_paper"
      app_role: "student" | "supervisor" | "admin"
      diagnostic_level: "beginner" | "developing" | "advanced"
      research_stage:
        | "topic_proposal"
        | "literature_review"
        | "methodology"
        | "data_analysis"
        | "results_writing"
        | "discussion"
        | "defense_preparation"
    }
    CompositeTypes: {
      [_ in never]: never
    }
  }
}

type DatabaseWithoutInternals = Omit<Database, "__InternalSupabase">

type DefaultSchema = DatabaseWithoutInternals[Extract<keyof Database, "public">]

export type Tables<
  DefaultSchemaTableNameOrOptions extends
    | keyof (DefaultSchema["Tables"] & DefaultSchema["Views"])
    | { schema: keyof DatabaseWithoutInternals },
  TableName extends DefaultSchemaTableNameOrOptions extends {
    schema: keyof DatabaseWithoutInternals
  }
    ? keyof (DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Tables"] &
        DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Views"])
    : never = never,
> = DefaultSchemaTableNameOrOptions extends {
  schema: keyof DatabaseWithoutInternals
}
  ? (DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Tables"] &
      DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Views"])[TableName] extends {
      Row: infer R
    }
    ? R
    : never
  : DefaultSchemaTableNameOrOptions extends keyof (DefaultSchema["Tables"] &
        DefaultSchema["Views"])
    ? (DefaultSchema["Tables"] &
        DefaultSchema["Views"])[DefaultSchemaTableNameOrOptions] extends {
        Row: infer R
      }
      ? R
      : never
    : never

export type TablesInsert<
  DefaultSchemaTableNameOrOptions extends
    | keyof DefaultSchema["Tables"]
    | { schema: keyof DatabaseWithoutInternals },
  TableName extends DefaultSchemaTableNameOrOptions extends {
    schema: keyof DatabaseWithoutInternals
  }
    ? keyof DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Tables"]
    : never = never,
> = DefaultSchemaTableNameOrOptions extends {
  schema: keyof DatabaseWithoutInternals
}
  ? DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Tables"][TableName] extends {
      Insert: infer I
    }
    ? I
    : never
  : DefaultSchemaTableNameOrOptions extends keyof DefaultSchema["Tables"]
    ? DefaultSchema["Tables"][DefaultSchemaTableNameOrOptions] extends {
        Insert: infer I
      }
      ? I
      : never
    : never

export type TablesUpdate<
  DefaultSchemaTableNameOrOptions extends
    | keyof DefaultSchema["Tables"]
    | { schema: keyof DatabaseWithoutInternals },
  TableName extends DefaultSchemaTableNameOrOptions extends {
    schema: keyof DatabaseWithoutInternals
  }
    ? keyof DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Tables"]
    : never = never,
> = DefaultSchemaTableNameOrOptions extends {
  schema: keyof DatabaseWithoutInternals
}
  ? DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Tables"][TableName] extends {
      Update: infer U
    }
    ? U
    : never
  : DefaultSchemaTableNameOrOptions extends keyof DefaultSchema["Tables"]
    ? DefaultSchema["Tables"][DefaultSchemaTableNameOrOptions] extends {
        Update: infer U
      }
      ? U
      : never
    : never

export type Enums<
  DefaultSchemaEnumNameOrOptions extends
    | keyof DefaultSchema["Enums"]
    | { schema: keyof DatabaseWithoutInternals },
  EnumName extends DefaultSchemaEnumNameOrOptions extends {
    schema: keyof DatabaseWithoutInternals
  }
    ? keyof DatabaseWithoutInternals[DefaultSchemaEnumNameOrOptions["schema"]]["Enums"]
    : never = never,
> = DefaultSchemaEnumNameOrOptions extends {
  schema: keyof DatabaseWithoutInternals
}
  ? DatabaseWithoutInternals[DefaultSchemaEnumNameOrOptions["schema"]]["Enums"][EnumName]
  : DefaultSchemaEnumNameOrOptions extends keyof DefaultSchema["Enums"]
    ? DefaultSchema["Enums"][DefaultSchemaEnumNameOrOptions]
    : never

export type CompositeTypes<
  PublicCompositeTypeNameOrOptions extends
    | keyof DefaultSchema["CompositeTypes"]
    | { schema: keyof DatabaseWithoutInternals },
  CompositeTypeName extends PublicCompositeTypeNameOrOptions extends {
    schema: keyof DatabaseWithoutInternals
  }
    ? keyof DatabaseWithoutInternals[PublicCompositeTypeNameOrOptions["schema"]]["CompositeTypes"]
    : never = never,
> = PublicCompositeTypeNameOrOptions extends {
  schema: keyof DatabaseWithoutInternals
}
  ? DatabaseWithoutInternals[PublicCompositeTypeNameOrOptions["schema"]]["CompositeTypes"][CompositeTypeName]
  : PublicCompositeTypeNameOrOptions extends keyof DefaultSchema["CompositeTypes"]
    ? DefaultSchema["CompositeTypes"][PublicCompositeTypeNameOrOptions]
    : never

export const Constants = {
  public: {
    Enums: {
      academic_track: [
        "undergraduate_project",
        "msc_thesis",
        "phd_research",
        "research_paper",
      ],
      app_role: ["student", "supervisor", "admin"],
      diagnostic_level: ["beginner", "developing", "advanced"],
      research_stage: [
        "topic_proposal",
        "literature_review",
        "methodology",
        "data_analysis",
        "results_writing",
        "discussion",
        "defense_preparation",
      ],
    },
  },
} as const
