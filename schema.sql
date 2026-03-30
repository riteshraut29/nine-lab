-- ════════════════════════════════════════════════════════════════════════════
--  NINE LAB 2.0 — Supabase Schema
--  Run this in the Supabase SQL editor (Dashboard → SQL Editor → New query)
-- ════════════════════════════════════════════════════════════════════════════

-- ── 1. User Profiles ─────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS user_profiles (
  id                  UUID        PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
  email               TEXT        NOT NULL DEFAULT '',
  full_name           TEXT        DEFAULT '',
  phone               TEXT        DEFAULT '',
  linkedin_url        TEXT        DEFAULT '',
  city                TEXT        DEFAULT '',
  degree              TEXT        DEFAULT '',
  field_of_study      TEXT        DEFAULT '',
  college             TEXT        DEFAULT '',
  grad_year           TEXT        DEFAULT '',
  cgpa                TEXT        DEFAULT '',
  is_fresher          BOOLEAN     DEFAULT TRUE,
  skills              TEXT        DEFAULT '',
  projects            TEXT        DEFAULT '',
  achievements        TEXT        DEFAULT '',
  roles               JSONB       DEFAULT '[]',
  profile_complete    BOOLEAN     DEFAULT FALSE,
  profile_pct         INTEGER     DEFAULT 0,
  created_at          TIMESTAMPTZ DEFAULT NOW(),
  updated_at          TIMESTAMPTZ DEFAULT NOW()
);

-- Auto-update timestamp
CREATE OR REPLACE FUNCTION set_updated_at()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_user_profiles_updated
  BEFORE UPDATE ON user_profiles
  FOR EACH ROW EXECUTE FUNCTION set_updated_at();

-- Auto-create stub on new user signup
CREATE OR REPLACE FUNCTION handle_new_user()
RETURNS TRIGGER AS $$
BEGIN
  INSERT INTO public.user_profiles (id, email)
  VALUES (NEW.id, COALESCE(NEW.email, ''))
  ON CONFLICT (id) DO NOTHING;
  RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

CREATE OR REPLACE TRIGGER trg_new_user_profile
  AFTER INSERT ON auth.users
  FOR EACH ROW EXECUTE FUNCTION handle_new_user();

-- RLS
ALTER TABLE user_profiles ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "profiles_own" ON user_profiles;
CREATE POLICY "profiles_own" ON user_profiles
  FOR ALL USING (auth.uid() = id);


-- ── 2. Opportunities ─────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS opportunities (
  id              UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
  title           TEXT        NOT NULL,
  company         TEXT        NOT NULL DEFAULT '',
  type            TEXT        NOT NULL DEFAULT 'job'
                    CHECK (type IN ('job', 'internship', 'hackathon', 'opensource', 'scholarship')),
  location        TEXT        DEFAULT 'India',
  url             TEXT        DEFAULT '',
  description     TEXT        DEFAULT '',
  requirements    TEXT        DEFAULT '',
  salary_range    TEXT        DEFAULT '',
  tags            TEXT[]      DEFAULT '{}',
  match_fields    TEXT        DEFAULT '',   -- space-joined keywords for server-side ranking
  deadline        DATE,
  is_active       BOOLEAN     DEFAULT TRUE,
  created_at      TIMESTAMPTZ DEFAULT NOW(),
  updated_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_opportunities_type ON opportunities(type, is_active);
CREATE INDEX IF NOT EXISTS idx_opportunities_active ON opportunities(is_active, created_at DESC);

-- RLS — all authenticated users can read, only service role can write
ALTER TABLE opportunities ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "opps_read" ON opportunities;
CREATE POLICY "opps_read" ON opportunities
  FOR SELECT USING (is_active = TRUE);


-- ── 3. Applications ───────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS applications (
  id                  UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id             UUID        NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  opportunity_id      UUID        REFERENCES opportunities(id) ON DELETE SET NULL,
  custom_title        TEXT        DEFAULT '',  -- for jobs not in opportunities table
  custom_company      TEXT        DEFAULT '',
  custom_url          TEXT        DEFAULT '',
  status              TEXT        NOT NULL DEFAULT 'saved'
                        CHECK (status IN ('saved','applied','interviewing','offered','rejected','withdrawn')),
  resume_generated    BOOLEAN     DEFAULT FALSE,
  prep_downloaded     BOOLEAN     DEFAULT FALSE,
  redirect_clicked    BOOLEAN     DEFAULT FALSE,
  applied_at          TIMESTAMPTZ,
  notes               TEXT        DEFAULT '',
  created_at          TIMESTAMPTZ DEFAULT NOW(),
  updated_at          TIMESTAMPTZ DEFAULT NOW()
);

CREATE TRIGGER trg_applications_updated
  BEFORE UPDATE ON applications
  FOR EACH ROW EXECUTE FUNCTION set_updated_at();

CREATE INDEX IF NOT EXISTS idx_applications_user ON applications(user_id, status);
CREATE INDEX IF NOT EXISTS idx_applications_user_opp ON applications(user_id, opportunity_id);

-- RLS
ALTER TABLE applications ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "apps_own" ON applications;
CREATE POLICY "apps_own" ON applications
  FOR ALL USING (auth.uid() = user_id);


-- ── 4. Activity Logs ─────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS activity_logs (
  id          UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id     UUID        NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  action_type TEXT        NOT NULL,
  metadata    JSONB       DEFAULT '{}',
  created_at  TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_activity_user ON activity_logs(user_id, created_at DESC);

ALTER TABLE activity_logs ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "activity_insert" ON activity_logs;
CREATE POLICY "activity_insert" ON activity_logs
  FOR INSERT WITH CHECK (auth.uid() = user_id);

DROP POLICY IF EXISTS "activity_select" ON activity_logs;
CREATE POLICY "activity_select" ON activity_logs
  FOR SELECT USING (auth.uid() = user_id);


-- ════════════════════════════════════════════════════════════════════════════
--  Done. Run this once in the Supabase SQL editor.
--  Then call POST /ninelab/admin/seed-opportunities to populate the feed.
-- ════════════════════════════════════════════════════════════════════════════
