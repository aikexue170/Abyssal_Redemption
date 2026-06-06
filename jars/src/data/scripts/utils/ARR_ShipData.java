package data.scripts.utils;

import com.fs.starfarer.api.combat.ShipAPI;
import org.lwjgl.util.vector.Vector2f;

public class ARR_ShipData {

    public final ShipAPI ship;
    public final String shipId;

    private Vector2f location;
    private Vector2f velocity;
    private float angle;
    private float angularVelocity;

    public ARR_ShipData(ShipAPI ship) {
        this.ship = ship;
        this.shipId = ship.getId();
    }

    public void update(Vector2f location, Vector2f velocity, float angle, float angularVelocity){
        this.location = location;
        this.velocity = velocity;
        this.angle = angle;
        this.angularVelocity = angularVelocity;
    }

    public Vector2f getLocation(){
        return this.location;
    }

    public Vector2f getVelocity(){
        return this.velocity;
    }

    public float getAngle(){
        return this.angle;
    }

    public float getAngularVelocity(){
        return this.angularVelocity;
    }
}
