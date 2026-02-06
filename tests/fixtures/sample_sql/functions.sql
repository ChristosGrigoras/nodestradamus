CREATE OR REPLACE FUNCTION public.get_user_orders(p_user_id integer)
RETURNS integer
LANGUAGE plpgsql
AS $$
BEGIN
    PERFORM public.log_access(p_user_id);
    INSERT INTO public.orders(user_id, amount) VALUES (p_user_id, 0);
    RETURN 1;
END;
$$;

CREATE OR REPLACE FUNCTION public.log_access(p_user_id integer)
RETURNS void
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN;
END;
$$;

CREATE OR REPLACE PROCEDURE public.cleanup_orders()
LANGUAGE plpgsql
AS $$
BEGIN
    DELETE FROM public.orders;
END;
$$;

CREATE TRIGGER audit_orders
AFTER INSERT ON public.orders
FOR EACH ROW EXECUTE FUNCTION public.log_access(NEW.user_id);
